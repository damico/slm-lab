package br.com.slmlab.avaliador.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("avaliador")
public record AvaliadorProperties(String inferenceUrl, String openaiApiKey,
        String openaiModel, String openaiBaseUrl, double inputPricePerMillion,
        double outputPricePerMillion) {
    public boolean comparisonEnabled() {
        return openaiApiKey != null && !openaiApiKey.isBlank();
    }
}
