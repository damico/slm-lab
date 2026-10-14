package br.com.slmlab.avaliador.controller;
import java.util.Map;
import org.springframework.http.*;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
@RestControllerAdvice(assignableTypes=ToolsController.class)
public class ToolsErrors {
    @ExceptionHandler({MethodArgumentNotValidException.class,HttpMessageNotReadableException.class})
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String,String> invalid(Exception error){return Map.of("message","Confira a solicitação e o catálogo JSON de 1 a 64 ferramentas. Não envie gabaritos.");}
    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<Map<String,String>> unavailable(ResponseStatusException error){return ResponseEntity.status(error.getStatusCode()).body(Map.of("message",error.getReason()));}
}
