package br.com.slmlab.avaliador.service;

import br.com.slmlab.avaliador.config.AvaliadorProperties;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.Message;
import java.util.ArrayList;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.stereotype.Service;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

@Service
public class ComparisonService {
    private final AvaliadorProperties properties;
    private final ChatClient client;

    public ComparisonService(AvaliadorProperties properties) {
        this.properties = properties;
        if (properties.comparisonEnabled()) {
            var api = OpenAiApi.builder().apiKey(properties.openaiApiKey())
                    .baseUrl(properties.openaiBaseUrl()).build();
            var model = OpenAiChatModel.builder().openAiApi(api)
                    .defaultOptions(OpenAiChatOptions.builder().model(properties.openaiModel())
                            .temperature(0.0).maxTokens(32).build())
                    .retryTemplate(RetryTemplate.builder().maxAttempts(1).build()).build();
            client = ChatClient.builder(model).build();
        } else {
            client = null;
        }
    }


    public Map<String, Object> compareSql(JsonNode context) {
        if (!enabled()) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Comparação OpenAI não configurada.");
        var messages = new ArrayList<Message>();
        for (JsonNode message : context.path("messages")) {
            String text = message.path("content").asText();
            messages.add("system".equals(message.path("role").asText()) ? new SystemMessage(text) : new UserMessage(text));
        }
        long start = System.nanoTime();
        var response = client.prompt().messages(messages)
                .options(OpenAiChatOptions.builder().model(properties.openaiModel()).temperature(0.0)
                        .maxTokens(context.path("max_new").asInt(512)).build()).call().chatResponse();
        if (response == null || response.getResult() == null)
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "OpenAI retornou uma resposta vazia.");
        var usage = response.getMetadata().getUsage();
        int input = usage.getPromptTokens() == null ? 0 : usage.getPromptTokens();
        int output = usage.getCompletionTokens() == null ? 0 : usage.getCompletionTokens();
        String sql = response.getResult().getOutput().getText();
        var result = new LinkedHashMap<String,Object>();
        result.put("model", properties.openaiModel());
        result.put("sql", sql == null ? "" : sql.strip());
        result.put("status", "length".equalsIgnoreCase(response.getResult().getMetadata().getFinishReason()) ? "output_limit" : "ok");
        result.put("latency_s", (System.nanoTime()-start)/1e9);
        result.put("prompt_tokens", input); result.put("completion_tokens", output);
        result.put("estimated_cost_usd", (input*properties.inputPricePerMillion()+output*properties.outputPricePerMillion())/1_000_000);
        return result;
    }

    public Map<String,Object> compareTools(JsonNode context) {
        var result=compareSql(context);
        result.put("text",result.remove("sql"));
        return result;
    }

    public Map<String,Object> compareReceipts(JsonNode context) {
        return compareTools(context);
    }

    public boolean enabled() { return client != null; }

    public Map<String, Object> compare(JsonNode context) {
        if (!enabled()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                    "Comparação OpenAI não configurada neste servidor.");
        }
        long start = System.nanoTime();
        var response = client.prompt().messages(new UserMessage(context.path("prompt").asText()))
                .call().chatResponse();
        if (response == null || response.getResult() == null) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "OpenAI retornou uma resposta vazia.");
        }
        var output = response.getResult().getOutput().getText();
        String choice = output == null ? "" : output.strip();
        String finish = response.getResult().getMetadata().getFinishReason();
        boolean valid = false;
        for (JsonNode label : context.path("labels")) {
            if (label.asText().equals(choice)) { valid = true; break; }
        }
        if ("length".equalsIgnoreCase(finish)) { valid = false; }
        var usage = response.getMetadata().getUsage();
        int input = usage.getPromptTokens() == null ? 0 : usage.getPromptTokens();
        int outputTokens = usage.getCompletionTokens() == null ? 0 : usage.getCompletionTokens();
        var result = new LinkedHashMap<String, Object>();
        result.put("model", properties.openaiModel());
        result.put("choice", choice);
        result.put("status", valid ? "ok" : "invalid_output");
        result.put("latency_s", (System.nanoTime() - start) / 1e9);
        result.put("prompt_tokens", input);
        result.put("completion_tokens", outputTokens);
        result.put("estimated_cost_usd", (input * properties.inputPricePerMillion()
                + outputTokens * properties.outputPricePerMillion()) / 1_000_000);
        return result;
    }
}
