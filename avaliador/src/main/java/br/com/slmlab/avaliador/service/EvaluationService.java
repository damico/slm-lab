package br.com.slmlab.avaliador.service;

import br.com.slmlab.avaliador.model.DecisionRequest;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiConsumer;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

@Service
public class EvaluationService {
    private final InferenceService inference;
    private final ComparisonService comparison;
    public EvaluationService(InferenceService inference, ComparisonService comparison) {
        this.inference = inference;
        this.comparison = comparison;
    }

    public Map<String, Object> evaluate(DecisionRequest request, BiConsumer<String, Object> events) {
        if (request.compare() && !comparison.enabled()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Configure OPENAI_API_KEY para comparar.");
        }
        long start = System.nanoTime();
        events.accept("stage", Map.of("message", "Recuperando regras e montando o contexto"));
        JsonNode context = inference.post("/context", request);
        events.accept("context", context);
        events.accept("stage", Map.of("message", "Executando o SLM especializado"));
        JsonNode decision = inference.post("/decide", request);
        events.accept("slm", decision);
        var result = new LinkedHashMap<String, Object>();
        result.put("context", context);
        result.put("slm", decision);
        if (request.compare() && context.path("eligible").asBoolean()) {
            events.accept("stage", Map.of("message", "Comparando o mesmo prompt com OpenAI"));
            try {
                var answer = comparison.compare(context);
                result.put("comparison", answer);
                events.accept("comparison", answer);
            } catch (Exception error) {
                var failure = Map.of("status", "api_error", "message",
                        "A comparação OpenAI falhou. Confira chave, saldo e modelo no servidor.");
                result.put("comparison", failure);
                events.accept("comparison", failure);
            }
        }
        result.put("elapsed_s", (System.nanoTime() - start) / 1e9);
        return result;
    }
}
