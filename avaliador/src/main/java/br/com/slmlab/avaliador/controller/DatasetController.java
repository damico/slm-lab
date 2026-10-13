package br.com.slmlab.avaliador.controller;

import br.com.slmlab.avaliador.service.DatasetService;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/datasets")
public class DatasetController {
    private final DatasetService datasets;
    public DatasetController(DatasetService datasets) {this.datasets=datasets;}
    @GetMapping public JsonNode catalog() {return datasets.catalog();}
    @GetMapping("/samples") public JsonNode samples(@RequestParam String dataset,
        @RequestParam(defaultValue="") String q,@RequestParam(defaultValue="1") int page,
        @RequestParam(defaultValue="20") int size,@RequestParam(defaultValue="all") String eligible,
        @RequestParam(defaultValue="false") boolean random) {return datasets.samples(dataset,q,page,size,eligible,random);}
    @GetMapping("/detail") public JsonNode detail(@RequestParam String dataset,@RequestParam int ordinal) {return datasets.detail(dataset,ordinal);}
    @GetMapping("/image") public ResponseEntity<byte[]> image(@RequestParam String dataset,@RequestParam int ordinal) {return datasets.image(dataset,ordinal);}
}
