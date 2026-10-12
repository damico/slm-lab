package br.com.slmlab.avaliador;

import static org.assertj.core.api.Assertions.assertThat;
import br.com.slmlab.avaliador.config.AvaliadorProperties;
import br.com.slmlab.avaliador.service.ComparisonService;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class ComparisonServiceTest {
    @Test void springAiPreservesTheExactPromptAndReportsUsage() throws Exception {
        var mapper = new ObjectMapper();
        var received = new AtomicReference<String>();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/chat/completions", exchange -> {
            received.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            var body = "{\"id\":\"test\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"retrieve_utterance\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":3,\"total_tokens\":103}}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        try {
            var properties = new AvaliadorProperties("", "test-key", "gpt-4.1-mini",
                    "http://127.0.0.1:" + server.getAddress().getPort(), .40, 1.60);
            var service = new ComparisonService(properties);
            String prompt = "literal JSON: {\"history\":[{\"text\":\"Hello\"}]}";
            var context = mapper.createObjectNode().put("prompt", prompt);
            context.putArray("labels").add("retrieve_utterance");
            var result = service.compare(context);
            assertThat(result.get("status")).isEqualTo("ok");
            assertThat(result.get("prompt_tokens")).isEqualTo(100);
            assertThat(result.get("completion_tokens")).isEqualTo(3);
            var request = mapper.readTree(received.get());
            assertThat(request.get("messages").size()).isEqualTo(1);
            assertThat(request.get("messages").get(0).get("content").asText()).isEqualTo(prompt);
            assertThat(request.get("messages").get(0).get("role").asText()).isEqualTo("user");
            assertThat(request.get("max_tokens").asInt()).isEqualTo(32);
            assertThat(request.get("temperature").asDouble()).isEqualTo(0);
            assertThat(request.has("tools")).isFalse();
        } finally { server.stop(0); }
    }
}
