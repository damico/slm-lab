package br.com.slmlab.avaliador.controller;

import br.com.slmlab.avaliador.config.AvaliadorProperties;
import br.com.slmlab.avaliador.model.DecisionRequest;
import br.com.slmlab.avaliador.service.EvaluationService;
import br.com.slmlab.avaliador.service.InferenceService;
import jakarta.validation.Valid;
import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api")
public class DecisionController {
    private final InferenceService inference;
    private final EvaluationService evaluation;
    private final AvaliadorProperties properties;
    private final ThreadPoolTaskExecutor executor;

    public DecisionController(InferenceService inference, EvaluationService evaluation,
            AvaliadorProperties properties, ThreadPoolTaskExecutor inferenceExecutor) {
        this.inference = inference;
        this.evaluation = evaluation;
        this.properties = properties;
        this.executor = inferenceExecutor;
    }

    @GetMapping("/status")
    public Map<String, Object> status() {
        var result = new LinkedHashMap<String, Object>();
        result.put("comparison_enabled", properties.comparisonEnabled());
        result.put("comparison_model", properties.openaiModel());
        try {
            result.put("inference", inference.get("/health"));
            result.put("available", true);
        } catch (ResponseStatusException error) {
            result.put("available", false);
            result.put("message", "Aguardando a API Python do Lab 04.");
        }
        return result;
    }

    @GetMapping("/lab04/catalog")
    public JsonNode catalog() { return inference.get("/catalog"); }

    @GetMapping("/lab04/examples")
    public JsonNode examples() { return inference.get("/examples"); }

    @PostMapping("/lab04/predict")
    public Map<String, Object> predict(@Valid @RequestBody DecisionRequest request) {
        return evaluation.evaluate(request, (name, value) -> {});
    }

    @PostMapping(value = "/lab04/predict/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter stream(@Valid @RequestBody DecisionRequest request) {
        var emitter = new SseEmitter(180_000L);
        try {
            executor.execute(() -> {
                try {
                    var result = evaluation.evaluate(request, (name, data) -> send(emitter, name, data));
                    send(emitter, "done", result);
                    emitter.complete();
                } catch (Exception error) {
                    String message = error instanceof ResponseStatusException status
                            ? status.getReason() : "Não foi possível concluir a inferência. Confira o serviço Python.";
                    try { send(emitter, "failure", Map.of("message", message)); } catch (Exception ignored) {}
                    emitter.complete();
                }
            });
        } catch (Exception busy) {
            send(emitter, "failure", Map.of("message", "Fila de testes cheia. Aguarde e tente novamente."));
            emitter.complete();
        }
        return emitter;
    }

    private static void send(SseEmitter emitter, String event, Object data) {
        try {
            emitter.send(SseEmitter.event().name(event).data(data, MediaType.APPLICATION_JSON));
        } catch (IOException error) {
            throw new IllegalStateException("Conexão com o navegador encerrada.", error);
        }
    }
}
