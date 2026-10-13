package br.com.slmlab.avaliador.service;

import com.fasterxml.jackson.databind.JsonNode;
import java.time.Duration;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.util.UriComponentsBuilder;

@Service
public class DatasetService {
    private final RestClient client;
    private final java.net.URI base;
    public DatasetService(@Value("${avaliador.dataset-url}") String url) {
        base=java.net.URI.create(url);
        var factory=new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(3));factory.setReadTimeout(Duration.ofSeconds(30));
        client=RestClient.builder().baseUrl(url).requestFactory(factory).build();
    }
    public JsonNode catalog() { return get("/catalog"); }
    public JsonNode samples(String dataset,String query,int page,int size,String eligible,boolean random) {
        return get(UriComponentsBuilder.fromPath("/samples").queryParam("dataset",dataset).queryParam("q",query)
            .queryParam("page",page).queryParam("size",size).queryParam("eligible",eligible).queryParam("random",random)
            .build().encode().toUriString());
    }
    public JsonNode detail(String dataset,int ordinal) { return get(path("/detail",dataset,ordinal)); }
    public ResponseEntity<byte[]> image(String dataset,int ordinal) {
        try {
            var upstream=client.get().uri(base.resolve(path("/image",dataset,ordinal))).retrieve().toEntity(byte[].class);
            return ResponseEntity.ok().contentType(upstream.getHeaders().getContentType())
                .header("X-Content-Type-Options","nosniff").body(upstream.getBody());
        } catch(RestClientException error) { throw translate(error); }
    }
    private String path(String path,String dataset,int ordinal) {
        return UriComponentsBuilder.fromPath(path).queryParam("dataset",dataset).queryParam("ordinal",ordinal)
            .build().encode().toUriString();
    }
    private JsonNode get(String path) {
        try { return client.get().uri(base.resolve(path)).retrieve().body(JsonNode.class); }
        catch(RestClientException error) { throw translate(error); }
    }
    private ResponseStatusException translate(RestClientException error) {
        if(error instanceof RestClientResponseException response && response.getStatusCode().value()==400)
            return new ResponseStatusException(HttpStatus.BAD_REQUEST,"Não foi possível abrir este filtro ou exemplo. Atualize a lista e tente novamente.");
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,"Os datasets locais ainda não estão conectados nesta instalação.");
    }
}
