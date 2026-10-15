package br.com.slmlab.avaliador.controller;
import java.util.Map;
import org.springframework.http.*;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
@RestControllerAdvice(assignableTypes=FinqaController.class)
public class FinqaErrors {
    @ExceptionHandler({MethodArgumentNotValidException.class,HttpMessageNotReadableException.class})
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String,String> invalid(Exception error){return Map.of("message","Confira a pergunta, as fontes e a tabela. Não envie gabaritos.");}
    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<Map<String,String>> unavailable(ResponseStatusException error){return ResponseEntity.status(error.getStatusCode()).body(Map.of("message",error.getReason()));}
}
