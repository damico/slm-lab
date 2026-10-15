package br.com.slmlab.avaliador.model;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.*;
import java.util.LinkedHashMap;
import java.util.Map;
public record FinqaRequest(@NotNull JsonNode context,
        @JsonProperty("example_id") @Size(max=200) String exampleId,
        @Pattern(regexp="recorded|specialist") String mode) {
    public Map<String,Object> input(){
        var result=new LinkedHashMap<String,Object>();result.put("context",context);
        result.put("mode",mode==null?"recorded":mode);
        if(exampleId!=null)result.put("example_id",exampleId);
        return result;
    }
}
