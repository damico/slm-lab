package br.com.slmlab.avaliador.model;
import jakarta.validation.constraints.*;
public record ReceiptUpload(@NotBlank @Size(max=200) String filename,
        @NotBlank @Size(max=12000000) String content) {}
