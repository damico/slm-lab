package br.com.slmlab.avaliador.controller;

import br.com.slmlab.avaliador.model.Lab;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.server.ResponseStatusException;

@Controller
public class PageController {
    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("labs", Lab.all());
        model.addAttribute("active", "home");
        return "index";
    }

    @GetMapping("/labs/{id}")
    public String lab(@PathVariable String id, Model model) {
        Lab lab = find(id);
        model.addAttribute("lab", lab);
        model.addAttribute("active", id);
        return lab.supported() ? "lab04/decisions" : "pending";
    }

    @GetMapping("/docs/{id}")
    public ResponseEntity<ClassPathResource> document(@PathVariable String id) {
        return ResponseEntity.ok().header("Content-Type", "text/plain; charset=UTF-8")
                .body(new ClassPathResource("docs/" + find(id).document()));
    }

    private Lab find(String id) {
        return Lab.all().stream().filter(lab -> lab.id().equals(id)).findFirst()
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Laboratório inexistente."));
    }
}
