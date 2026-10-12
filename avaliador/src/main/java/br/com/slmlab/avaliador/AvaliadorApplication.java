package br.com.slmlab.avaliador;

import br.com.slmlab.avaliador.config.AvaliadorProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties(AvaliadorProperties.class)
public class AvaliadorApplication {
    public static void main(String[] args) {
        SpringApplication.run(AvaliadorApplication.class, args);
    }
}
