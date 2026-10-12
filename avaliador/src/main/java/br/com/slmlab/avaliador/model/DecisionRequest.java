package br.com.slmlab.avaliador.model;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonInclude;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record DecisionRequest(
        @NotEmpty @Size(max = 200) List<@Valid Event> history,
        boolean compare) {
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record Event(
            @NotBlank @Pattern(regexp = "customer|agent|action") String role,
            @NotBlank @Size(max = 16000) String text,
            @Size(max = 100) String action) {}
}
