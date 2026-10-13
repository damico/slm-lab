package br.com.slmlab.avaliador.model;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

public record SqlRequest(@JsonProperty("db_id") @NotBlank @Size(max=100) String dbId,
        @NotBlank @Size(max=8000) String question, @NotNull @Size(max=12000) String evidence, boolean compare) {
    public Map<String, String> input() { return Map.of("db_id", dbId, "question", question, "evidence", evidence); }
}
