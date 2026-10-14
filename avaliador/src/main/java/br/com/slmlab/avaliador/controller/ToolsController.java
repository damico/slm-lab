package br.com.slmlab.avaliador.controller;

import br.com.slmlab.avaliador.model.ToolsRequest;
import br.com.slmlab.avaliador.service.ToolsInferenceService;
import br.com.slmlab.avaliador.service.ComparisonService;
import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.Valid;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api/lab01")
public class ToolsController {
    private final ToolsInferenceService inference;
    private final ComparisonService comparison;
    private final ThreadPoolTaskExecutor executor;
    public ToolsController(ToolsInferenceService inference,ComparisonService comparison,ThreadPoolTaskExecutor inferenceExecutor) {
        this.inference=inference;this.comparison=comparison;this.executor=inferenceExecutor;
    }
    @GetMapping("/status") public Map<String,Object> status() {
        var state=new LinkedHashMap<String,Object>();state.put("comparison_enabled",comparison.enabled());
        try {state.put("inference",inference.get("/health"));state.put("available",true);}
        catch(ResponseStatusException error){state.put("available",false);state.put("message",error.getReason());}
        return state;
    }
    @GetMapping("/catalog") public JsonNode catalog(){return inference.get("/catalog");}
    @GetMapping("/example/random") public JsonNode random(){return inference.get("/example/random");}
    private void check(ToolsRequest input) {
        if(input.compare()&&!comparison.enabled())throw new ResponseStatusException(HttpStatus.BAD_REQUEST,"Comparação OpenAI não configurada.");
    }
    private Map<String,Object> evaluate(ToolsRequest input) {
        long start=System.nanoTime();JsonNode prediction=inference.post("/predict",input.input());
        var result=new LinkedHashMap<String,Object>();prediction.fields().forEachRemaining(f->result.put(f.getKey(),f.getValue()));
        if(input.compare()&& !"context_exceeded".equals(prediction.path("slm").path("status").asText())) {
            try {
                var api=comparison.compareTools(prediction.path("context"));
                api.put("validation",inference.post("/validate",Map.of("text",api.get("text"),"tools",input.tools(),"status",api.get("status"))));
                result.put("comparison",api);
            } catch(Exception error){result.put("comparison",Map.of("status","api_error","message","Comparação indisponível; resultado local preservado."));}
        }
        result.put("elapsed_s",(System.nanoTime()-start)/1e9);return result;
    }
    @PostMapping("/predict") public Map<String,Object> predict(@Valid @RequestBody ToolsRequest input){check(input);return evaluate(input);}
    @PostMapping(value="/predict/stream",produces=MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@Valid @RequestBody ToolsRequest input) {
        check(input);var emitter=new SseEmitter(180000L);
        try {executor.execute(()->{
            try {send(emitter,"stage",Map.of("message","specialist".equals(input.mode())?"Carregando o especialista e propondo chamadas…":"Consultando a execução registrada deste exemplo…"));
                send(emitter,"done",evaluate(input));emitter.complete();}
            catch(Exception error){send(emitter,"failure",Map.of("message",error instanceof ResponseStatusException s?s.getReason():"Falha no teste. Confira o log da API de ferramentas."));emitter.complete();}
        });}catch(Exception error){send(emitter,"failure",Map.of("message","Fila cheia. Aguarde e tente novamente."));emitter.complete();}
        return emitter;
    }
    private static void send(SseEmitter emitter,String event,Object data) {
        try {emitter.send(SseEmitter.event().name(event).data(data,MediaType.APPLICATION_JSON));}
        catch(IOException error){throw new IllegalStateException(error);}
    }
}
