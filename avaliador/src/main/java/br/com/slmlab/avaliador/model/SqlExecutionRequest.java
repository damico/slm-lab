package br.com.slmlab.avaliador.model;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
public record SqlExecutionRequest(@JsonProperty("db_id") @NotBlank @Size(max=100) String dbId,
        @NotBlank @Size(max=16000) String sql) {}
