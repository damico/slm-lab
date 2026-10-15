package br.com.slmlab.avaliador.service;

import br.com.slmlab.avaliador.model.Lab;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.commonmark.ext.gfm.tables.TablesExtension;
import org.commonmark.node.AbstractVisitor;
import org.commonmark.node.Link;
import org.commonmark.parser.Parser;
import org.commonmark.renderer.html.HtmlRenderer;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

@Service
public class DocumentService {
    private final Parser parser = Parser.builder().extensions(List.of(TablesExtension.create())).build();
    private final HtmlRenderer renderer = HtmlRenderer.builder().extensions(List.of(TablesExtension.create()))
            .escapeHtml(true).sanitizeUrls(true).build();

    public String render(Lab lab) throws IOException {
        String source = new ClassPathResource("docs/" + lab.document()).getContentAsString(StandardCharsets.UTF_8);
        var document = parser.parse(source);
        document.accept(new AbstractVisitor() {
            @Override public void visit(Link link) {
                String destination = link.getDestination();
                if (!destination.startsWith("#") && !destination.startsWith("/") && !destination.contains(":")) {
                    String path = destination.split("#", 2)[0];
                    String anchor = destination.substring(path.length());
                    String internal = Lab.all().stream().filter(item -> item.document().equals(path))
                            .map(item -> "/docs/" + item.id() + anchor).findFirst().orElse(null);
                    link.setDestination(internal != null ? internal : "https://github.com/damico/slm-lab/blob/main/" + destination);
                }
                super.visit(link);
            }
        });
        return renderer.render(document);
    }
}
