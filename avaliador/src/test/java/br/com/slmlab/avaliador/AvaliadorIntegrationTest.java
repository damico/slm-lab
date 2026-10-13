package br.com.slmlab.avaliador;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.assertj.core.api.Assertions.assertThat;

import com.sun.net.httpserver.HttpServer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class AvaliadorIntegrationTest {
    static final HttpServer upstream;
    static final AtomicReference<String> lastQuery = new AtomicReference<>();
    static final AtomicReference<String> lastHistory = new AtomicReference<>();
    static final ObjectMapper mapper = new ObjectMapper();
    static {
        try {
            upstream = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            upstream.createContext("/", exchange -> {
                String path = exchange.getRequestURI().getPath();
                lastQuery.set(exchange.getRequestURI().getRawQuery());
                if (exchange.getRequestMethod().equals("POST")) {
                    if (exchange.getRequestHeaders().getFirst("Content-Length") == null) {
                        exchange.sendResponseHeaders(400, -1);
                        exchange.close();
                        return;
                    }
                    lastHistory.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
                }
                String body = switch (path) {
                    case "/health" -> "{\"status\":\"ready\",\"classes\":31}";
                    case "/catalog" -> "{\"labels\":[\"retrieve_utterance\"],\"actions\":[],\"labs\":[{\"id\":\"01\"}]}";
                    case "/samples" -> "{\"total\":1,\"items\":[{\"ordinal\":0,\"identifier\":\"tools:1\"}]}";
                    case "/detail" -> "{\"id\":\"tools:1\",\"question\":\"Weather in Paris\"}";
                    case "/examples" -> "{\"examples\":[]}";
                    case "/context" -> "{\"prompt\":\"literal {history} prompt\",\"eligible\":true,\"input_tokens\":12,\"policies\":[],\"labels\":[\"retrieve_utterance\"]}";
                    case "/predict" -> "{\"context\":{\"eligible\":true},\"slm\":{\"sql\":\"SELECT 1\",\"status\":\"ok\"}}";
                    case "/execute" -> "{\"status\":\"ok\",\"rows\":[[1]],\"columns\":[\"1\"]}";
                    case "/decide" -> "{\"choice\":\"retrieve_utterance\",\"confidence\":0.8,\"requires_review\":false,\"probabilities\":{\"retrieve_utterance\":0.8}}";
                    default -> "{}";
                };
                byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().add("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, bytes.length);
                exchange.getResponseBody().write(bytes);
                exchange.close();
            });
            upstream.start();
        } catch (IOException error) { throw new ExceptionInInitializerError(error); }
    }
    @DynamicPropertySource
    static void settings(DynamicPropertyRegistry registry) {
        registry.add("avaliador.inference-url", () -> "http://127.0.0.1:" + upstream.getAddress().getPort());
        registry.add("avaliador.sql-inference-url", () -> "http://127.0.0.1:" + upstream.getAddress().getPort());
        registry.add("avaliador.dataset-url", () -> "http://127.0.0.1:" + upstream.getAddress().getPort());
        registry.add("avaliador.openai-api-key", () -> "");
    }
    @Autowired MockMvc mvc;

    @AfterAll static void stop() { upstream.stop(0); }

    @Test void pagesAndLocalAssetsRenderWithoutApiKey() throws Exception {
        mvc.perform(get("/")).andExpect(status().isOk()).andExpect(content().string(org.hamcrest.Matchers.containsString("85,45")));
        mvc.perform(get("/labs/04")).andExpect(status().isOk()).andExpect(content().string(org.hamcrest.Matchers.containsString("timeline")));
        mvc.perform(get("/labs/02")).andExpect(status().isOk()).andExpect(content().string(org.hamcrest.Matchers.containsString("sql-question")));
        mvc.perform(get("/js/journey.js")).andExpect(status().isOk());
        mvc.perform(get("/labs/05")).andExpect(status().isOk()).andExpect(content().string(org.hamcrest.Matchers.containsString("finqa-question")));
        mvc.perform(get("/docs/05")).andExpect(status().isOk());
        mvc.perform(get("/docs/04")).andExpect(status().isOk());
        mvc.perform(get("/webjars/bootstrap/5.3.7/css/bootstrap.min.css")).andExpect(status().isOk());
        mvc.perform(get("/webjars/chart.js/4.5.1/dist/chart.umd.js")).andExpect(status().isOk());
        mvc.perform(get("/api/status")).andExpect(jsonPath("$.available").value(true)).andExpect(jsonPath("$.comparison_enabled").value(false));
    }

    @Test void sendsOnlyHistoryToTheModelAndPreservesItsOutput() throws Exception {
        mvc.perform(post("/api/lab04/predict").contentType("application/json")
            .content("{\"history\":[{\"role\":\"customer\",\"text\":\"Hello\"}],\"compare\":false}"))
            .andExpect(status().isOk()).andExpect(jsonPath("$.slm.choice").value("retrieve_utterance"))
            .andExpect(jsonPath("$.context.prompt").value("literal {history} prompt"));
        JsonNode payload = mapper.readTree(lastHistory.get());
        assertThat(payload.size()).isEqualTo(1);
        assertThat(payload.has("history")).isTrue();
        assertThat(payload.get("history").get(0).has("action")).isFalse();
    }

    @Test void rejectsInvalidInputsAndUnavailablePaidComparison() throws Exception {
        mvc.perform(post("/api/lab04/predict").contentType("application/json").content("{\"history\":[]}"))
            .andExpect(status().isBadRequest());
        mvc.perform(post("/api/lab04/predict").contentType("application/json")
            .content("{\"history\":[{\"role\":\"system\",\"text\":\"Hello\"}]}"))
            .andExpect(status().isBadRequest());
        mvc.perform(post("/api/lab04/predict").contentType("application/json")
            .content("{\"history\":[{\"role\":\"customer\",\"text\":\"Hello\"}],\"label\":\"answer\"}"))
            .andExpect(status().isBadRequest());
        mvc.perform(post("/api/lab04/predict").contentType("application/json")
            .content("{\"history\":[{\"role\":\"customer\",\"text\":\"Hello\"}],\"compare\":true}"))
            .andExpect(status().isBadRequest());
    }


    @Test void datasetExplorerIsIndependentAndEncodesSearchOnce() throws Exception {
        mvc.perform(get("/datasets")).andExpect(status().isOk())
            .andExpect(content().string(org.hamcrest.Matchers.containsString("Mostrar resposta esperada")));
        mvc.perform(get("/api/datasets")).andExpect(status().isOk()).andExpect(jsonPath("$.labs[0].id").value("01"));
        mvc.perform(get("/api/datasets/samples").param("dataset","01-prepared-train").param("q","Paris weather"))
            .andExpect(status().isOk()).andExpect(jsonPath("$.total").value(1));
        String query=java.net.URLDecoder.decode(lastQuery.get(),StandardCharsets.UTF_8);
        assertThat(query).contains("q=Paris weather");
        mvc.perform(get("/api/datasets/detail").param("dataset","01-prepared-train").param("ordinal","0"))
            .andExpect(status().isOk()).andExpect(jsonPath("$.question").value("Weather in Paris"));
    }

    @Test void sqlUsesSeparateContractAndRejectsGoldAndPaths() throws Exception {
        mvc.perform(get("/api/lab02/status")).andExpect(jsonPath("$.available").value(true));
        mvc.perform(post("/api/lab02/predict").contentType("application/json")
            .content("{\"db_id\":\"demo\",\"question\":\"How many?\",\"evidence\":\"\",\"compare\":false}"))
            .andExpect(status().isOk()).andExpect(jsonPath("$.slm.sql").value("SELECT 1"));
        JsonNode payload = mapper.readTree(lastHistory.get());
        assertThat(payload.size()).isEqualTo(3);
        assertThat(payload.has("compare")).isFalse();
        mvc.perform(post("/api/lab02/predict").contentType("application/json")
            .content("{\"db_id\":\"demo\",\"question\":\"How many?\",\"evidence\":\"\",\"gold\":\"SELECT 1\"}"))
            .andExpect(status().isBadRequest());
        mvc.perform(post("/api/lab02/execute").contentType("application/json")
            .content("{\"db_id\":\"demo\",\"sql\":\"SELECT 1\"}"))
            .andExpect(status().isOk()).andExpect(jsonPath("$.rows[0][0]").value(1));
    }

    @Test void sqlStreamReturnsCompletion() throws Exception {
        var result = mvc.perform(post("/api/lab02/predict/stream").contentType("application/json")
            .content("{\"db_id\":\"demo\",\"question\":\"List rows\",\"evidence\":\"\",\"compare\":false}"))
            .andExpect(request().asyncStarted()).andReturn();
        result.getAsyncResult(5000);
        mvc.perform(asyncDispatch(result)).andExpect(status().isOk())
            .andExpect(content().string(org.hamcrest.Matchers.containsString("event:done")));
    }

    @Test void sseDeliversContextDecisionAndCompletion() throws Exception {
        var result = mvc.perform(post("/api/lab04/predict/stream").contentType("application/json")
            .content("{\"history\":[{\"role\":\"customer\",\"text\":\"Hello\"}],\"compare\":false}"))
            .andExpect(request().asyncStarted()).andReturn();
        result.getAsyncResult(5000);
        mvc.perform(asyncDispatch(result)).andExpect(status().isOk())
            .andExpect(content().string(org.hamcrest.Matchers.containsString("event:context")))
            .andExpect(content().string(org.hamcrest.Matchers.containsString("event:slm")))
            .andExpect(content().string(org.hamcrest.Matchers.containsString("event:done")));
    }
}
