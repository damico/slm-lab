package br.com.slmlab.avaliador.controller;

import br.com.slmlab.avaliador.model.Lab;
import java.io.IOException;
import br.com.slmlab.avaliador.service.DocumentService;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.server.ResponseStatusException;

@Controller
public class PageController {
    private final DocumentService documents;

    public PageController(DocumentService documents) { this.documents = documents; }

    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("labs", Lab.all());
        model.addAttribute("active", "home");
        return "index";
    }

    @GetMapping("/datasets")
    public String datasets(Model model) {
        model.addAttribute("active", "datasets");
        return "datasets";
    }

    @GetMapping("/labs/{id}")
    public String lab(@PathVariable String id, Model model) {
        Lab lab = find(id);
        model.addAttribute("lab", lab);
        model.addAttribute("active", id);
        return switch (id) { case "01" -> "lab01/tools"; case "02" -> "lab02/sql"; case "03" -> "lab03/receipts"; case "04" -> "lab04/decisions"; default -> "pending"; };
    }

    @GetMapping("/docs/{id}")
    public String document(@PathVariable String id, Model model) throws IOException {
        Lab lab = find(id);
        model.addAttribute("lab", lab);
        model.addAttribute("active", id);
        model.addAttribute("documentHtml", documents.render(lab));
        return "document";
    }

    private Lab find(String id) {
        return Lab.all().stream().filter(lab -> lab.id().equals(id)).findFirst()
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Laboratório inexistente."));
    }
}
