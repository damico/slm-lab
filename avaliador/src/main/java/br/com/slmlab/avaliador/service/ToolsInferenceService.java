package br.com.slmlab.avaliador.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

@Service
public class ToolsInferenceService extends SqlInferenceService {
    public ToolsInferenceService(@Value("${avaliador.tools-inference-url}") String url,ObjectMapper mapper) {
        super(url,mapper);
    }
    @Override protected ResponseStatusException translate(RestClientException error) {
        if(error instanceof RestClientResponseException response && response.getStatusCode().value()==400)
            return new ResponseStatusException(HttpStatus.BAD_REQUEST,"Confira solicitação, catálogo e modo. Exemplos editados precisam da inferência real.");
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
            "API de ferramentas indisponível. Inicie avaliador/start-lab01.sh e confira LAB01_API_URL.");
    }
}
