package br.com.slmlab.avaliador.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Duration;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;

@Service
@org.springframework.context.annotation.Primary
public class SqlInferenceService {
    private final RestClient client;
    private final ObjectMapper mapper;
    public SqlInferenceService(@Value("${avaliador.sql-inference-url}") String url, ObjectMapper mapper) {
        this.mapper = mapper;
        var factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(3));
        factory.setReadTimeout(Duration.ofSeconds(120));
        client = RestClient.builder().baseUrl(url).requestFactory(factory).build();
    }
    public JsonNode get(String path) {
        try { return client.get().uri(path).retrieve().body(JsonNode.class); }
        catch (RestClientException error) { throw translate(error); }
    }
    public JsonNode post(String path, Map<String, ?> input) {
        try {
            byte[] body = mapper.writeValueAsBytes(input);
            return client.post().uri(path).contentType(MediaType.APPLICATION_JSON).contentLength(body.length)
                    .body(body).retrieve().body(JsonNode.class);
        } catch (com.fasterxml.jackson.core.JsonProcessingException error) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Entrada SQL inválida.");
        } catch (RestClientException error) { throw translate(error); }
    }
    protected ResponseStatusException translate(RestClientException error) {
        if (error instanceof RestClientResponseException response && response.getStatusCode().value() == 400)
            return new ResponseStatusException(HttpStatus.BAD_REQUEST, "Confira banco, pergunta e evidence.");
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                "API SQL indisponível. Inicie avaliador/start-lab02.sh e confira LAB02_API_URL.");
    }
}
