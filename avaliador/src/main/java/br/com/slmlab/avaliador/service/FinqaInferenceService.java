package br.com.slmlab.avaliador.service;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
@Service
public class FinqaInferenceService extends SqlInferenceService {
    public FinqaInferenceService(@Value("${avaliador.finqa-inference-url}") String url,ObjectMapper mapper){super(url,mapper);}
    public com.fasterxml.jackson.databind.JsonNode example(String id){
        try{return client.get().uri("/example?id={id}",id).retrieve().body(com.fasterxml.jackson.databind.JsonNode.class);}
        catch(RestClientException error){throw translate(error);}
    }
    @Override protected ResponseStatusException translate(RestClientException error){
        if(error instanceof RestClientResponseException response && response.getStatusCode().value()==400)
            return new ResponseStatusException(HttpStatus.BAD_REQUEST,"Confira pergunta, fontes e tabela. Entradas editadas precisam do modo de inferência real; não envie gabaritos.");
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
            "API FinQA indisponível. Inicie avaliador/start-lab05.sh e confira LAB05_API_URL.");
    }
}
