package br.com.slmlab.avaliador.service;

import br.com.slmlab.avaliador.model.DecisionRequest;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;

@Service
public class InferenceService {
    private final RestClient client;
    private final ObjectMapper mapper;
    public InferenceService(RestClient inferenceClient, ObjectMapper mapper) {
        this.client = inferenceClient;
        this.mapper = mapper;
    }

    public JsonNode get(String path) {
        try {
            return client.get().uri(path).retrieve().body(JsonNode.class);
        } catch (RestClientException error) {
            throw translate(error);
        }
    }

    public JsonNode post(String path, DecisionRequest request) {
        try {
            byte[] payload = mapper.writeValueAsBytes(Map.of("history", request.history()));
            return client.post().uri(path).contentType(MediaType.APPLICATION_JSON)
                    .contentLength(payload.length).body(payload)
                    .retrieve().body(JsonNode.class);
        } catch (JsonProcessingException error) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Não foi possível serializar o histórico.");
        } catch (RestClientException error) {
            throw translate(error);
        }
    }

    private ResponseStatusException translate(RestClientException error) {
        if (error instanceof RestClientResponseException response && response.getStatusCode().value() == 400) {
            return new ResponseStatusException(HttpStatus.BAD_REQUEST,
                    "Histórico recusado pelo serviço Python. Confira os papéis e nomes das ações.");
        }
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                "API do Lab 04 indisponível. Inicie serve_decisions.py ou confira LAB04_API_URL.");
    }
}
