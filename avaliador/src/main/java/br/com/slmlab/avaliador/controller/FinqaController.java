package br.com.slmlab.avaliador.controller;
import br.com.slmlab.avaliador.model.FinqaRequest;
import br.com.slmlab.avaliador.service.FinqaInferenceService;
import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.Valid;
import java.io.IOException;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
@RestController
@RequestMapping("/api/lab05")
public class FinqaController {
    private final FinqaInferenceService inference;
    private final ThreadPoolTaskExecutor executor;
    public FinqaController(FinqaInferenceService inference,ThreadPoolTaskExecutor inferenceExecutor){this.inference=inference;this.executor=inferenceExecutor;}
    @GetMapping("/status") public Map<String,Object> status(){
        try{return Map.of("available",true,"inference",inference.get("/health"));}
        catch(ResponseStatusException error){return Map.of("available",false,"message",error.getReason());}
    }
    @GetMapping("/catalog") public JsonNode catalog(){return inference.get("/catalog");}
    @GetMapping("/example/random") public JsonNode random(){return inference.get("/example/random");}
    @GetMapping("/example") public JsonNode example(@RequestParam String id){return inference.example(id);}
    @PostMapping("/predict") public JsonNode predict(@Valid @RequestBody FinqaRequest input){return inference.post("/predict",input.input());}
    @PostMapping(value="/predict/stream",produces=MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@Valid @RequestBody FinqaRequest input){
        var emitter=new SseEmitter(180000L);
        try{executor.execute(()->{
            try{send(emitter,"stage",Map.of("message","specialist".equals(input.mode())?"Gerando o programa com o aluno destilado e verificando o cálculo…":"Executando o programa registrado e conferindo as evidências…"));
                send(emitter,"done",inference.post("/predict",input.input()));emitter.complete();}
            catch(Exception error){send(emitter,"failure",Map.of("message",error instanceof ResponseStatusException s?s.getReason():"Falha no teste FinQA; consulte o log da API."));emitter.complete();}
        });}catch(Exception error){send(emitter,"failure",Map.of("message","Fila cheia. Aguarde e tente novamente."));emitter.complete();}
        return emitter;
    }
    private static void send(SseEmitter emitter,String event,Object data){
        try{emitter.send(SseEmitter.event().name(event).data(data,MediaType.APPLICATION_JSON));}
        catch(IOException error){throw new IllegalStateException(error);}
    }
}
