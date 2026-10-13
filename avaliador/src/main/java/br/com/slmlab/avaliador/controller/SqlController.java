package br.com.slmlab.avaliador.controller;

import br.com.slmlab.avaliador.model.SqlRequest;
import br.com.slmlab.avaliador.model.SqlExecutionRequest;
import br.com.slmlab.avaliador.service.SqlInferenceService;
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
@RequestMapping("/api/lab02")
public class SqlController {
    private final SqlInferenceService inference;
    private final ComparisonService comparison;
    private final ThreadPoolTaskExecutor executor;
    public SqlController(SqlInferenceService inference, ComparisonService comparison, ThreadPoolTaskExecutor inferenceExecutor) {
        this.inference = inference; this.comparison = comparison; this.executor = inferenceExecutor;
    }
    @GetMapping("/status") public Map<String, Object> status() {
        var state = new LinkedHashMap<String,Object>();
        state.put("comparison_enabled", comparison.enabled());
        try { state.put("inference", inference.get("/health")); state.put("available", true); }
        catch (ResponseStatusException error) { state.put("available", false); state.put("message", error.getReason()); }
        return state;
    }
    @GetMapping("/catalog") public JsonNode catalog() { return inference.get("/catalog"); }
    @PostMapping("/execute") public JsonNode execute(@Valid @RequestBody SqlExecutionRequest input) {
        return inference.post("/execute", Map.of("db_id", input.dbId(), "sql", input.sql()));
    }
    @PostMapping("/predict") public Map<String,Object> predict(@Valid @RequestBody SqlRequest input) {
        checkComparison(input);
        return evaluate(input);
    }
    private void checkComparison(SqlRequest input) {
        if (input.compare() && !comparison.enabled())
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Comparação OpenAI não configurada.");
    }
    private Map<String,Object> evaluate(SqlRequest input) {
        long start = System.nanoTime();
        JsonNode prediction = inference.post("/predict", input.input());
        var result = new LinkedHashMap<String,Object>();
        prediction.fields().forEachRemaining(field -> result.put(field.getKey(), field.getValue()));
        if (input.compare() && prediction.path("context").path("eligible").asBoolean()) {
            try {
                var api = comparison.compareSql(prediction.path("context"));
                if ("ok".equals(api.get("status")))
                    api.put("execution", inference.post("/execute", Map.of("db_id", input.dbId(), "sql", api.get("sql"))));
                result.put("comparison", api);
            } catch (Exception error) { result.put("comparison", Map.of("status", "api_error", "message", "Comparação indisponível; resultado local preservado.")); }
        }
        result.put("elapsed_s", (System.nanoTime()-start)/1e9);
        return result;
    }
    @PostMapping(value="/predict/stream", produces=MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@Valid @RequestBody SqlRequest input) {
        checkComparison(input);
        var emitter = new SseEmitter(180000L);
        try {
            executor.execute(() -> {
                try {
                    send(emitter, "stage", Map.of("message", "Gerando SQL com o adaptador treinado e executando a consulta…"));
                    var result = evaluate(input);
                    send(emitter, "done", result); emitter.complete();
                } catch (Exception error) {
                    try { send(emitter, "failure", Map.of("message", error instanceof ResponseStatusException status ? status.getReason() : "Falha no teste SQL. Confira o log Python.")); }
                    catch (Exception ignored) {}
                    emitter.complete();
                }
            });
        } catch (Exception error) {
            send(emitter, "failure", Map.of("message", "Fila cheia. Aguarde e tente novamente.")); emitter.complete();
        }
        return emitter;
    }
    private static void send(SseEmitter emitter, String event, Object data) {
        try { emitter.send(SseEmitter.event().name(event).data(data, MediaType.APPLICATION_JSON)); }
        catch (IOException error) { throw new IllegalStateException(error); }
    }
}
