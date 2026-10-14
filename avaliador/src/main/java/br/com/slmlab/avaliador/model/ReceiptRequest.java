package br.com.slmlab.avaliador.model;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.*;
import java.util.LinkedHashMap;
import java.util.Map;
public record ReceiptRequest(@NotNull @Size(max=20000) String text,
        @JsonProperty("example_id") @Size(max=120) String exampleId,
        @Pattern(regexp="recorded|specialist") String mode,boolean compare) {
    public Map<String,Object> input(){
        var result=new LinkedHashMap<String,Object>();result.put("text",text);
        result.put("mode",mode==null?"recorded":mode);
        if(exampleId!=null)result.put("example_id",exampleId);
        return result;
    }
}
