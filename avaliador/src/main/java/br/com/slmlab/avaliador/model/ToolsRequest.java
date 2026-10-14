package br.com.slmlab.avaliador.model;

import jakarta.validation.constraints.*;
import java.util.List;
import java.util.Map;

public record ToolsRequest(@NotBlank @Size(max=8000) String query,
        @NotNull @Size(min=1,max=64) List<Map<String,Object>> tools,
        @Pattern(regexp="recorded|specialist") String mode, boolean compare) {
    public Map<String,Object> input() {
        return Map.of("query",query,"tools",tools,"mode",mode==null?"recorded":mode);
    }
}
