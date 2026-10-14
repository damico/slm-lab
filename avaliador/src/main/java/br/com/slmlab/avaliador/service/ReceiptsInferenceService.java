package br.com.slmlab.avaliador.service;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.http.*;
import org.springframework.web.server.ResponseStatusException;
@Service
public class ReceiptsInferenceService extends SqlInferenceService {
    public ReceiptsInferenceService(@Value("${avaliador.receipts-inference-url}") String url,ObjectMapper mapper){super(url,mapper);}
    public ResponseEntity<byte[]> image(String id){
        try{return client.get().uri("/image?id={id}",id).retrieve().toEntity(byte[].class);}
        catch(RestClientException error){throw translate(error);}
    }
    @Override protected ResponseStatusException translate(RestClientException error){
        if(error instanceof RestClientResponseException response){
            if(response.getStatusCode().value()==400)return new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "Confira o texto, o modo e o arquivo. Entradas editadas precisam de inferência real; use imagem ou PDF de uma página, até 8 MB.");
            if(response.getStatusCode().value()==404)return new ResponseStatusException(HttpStatus.NOT_FOUND,"Imagem indisponível.");
        }
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
            "API de recibos indisponível. Inicie avaliador/start-lab03.sh e confira LAB03_API_URL.");
    }
}
