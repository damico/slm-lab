package br.com.slmlab.avaliador.model;

import java.util.List;

public record Lab(String id, String title, String category, String dataset, String description,
        String output, String document, boolean supported, List<String> steps) {
    public static List<Lab> all() {
        return List.of(
            new Lab("01", "Ferramentas", "Tool calling", "xLAM / Salesforce",
                "Transforme uma solicitação em chamadas de funções com argumentos estruturados.",
                "JSON com nome e argumentos das ferramentas", "laboratorio-01-slm-ferramentas-xlam.md", true,
                List.of("Solicitação + catálogo", "SLM especializado", "Chamadas em JSON")),
            new Lab("02", "SQL", "Text-to-SQL", "BIRD / Mini-Dev",
                "Faça perguntas a um banco e confira as respostas executando o SQL gerado.",
                "SQL e resultado da consulta", "laboratorio-02-slm-sql-bird.md", true,
                List.of("Pergunta + esquema + evidence", "SLM especializado", "SQL + avaliação de execução")),
            new Lab("03", "Recibos", "Extração estruturada", "SROIE / ICDAR2019",
                "Extraia empresa, data, endereço e total a partir de um recibo com OCR.",
                "JSON e CSV com quatro campos", "laboratorio-03-slm-extracao-recibos.md", true,
                List.of("Imagem ou PDF", "OCR / texto", "SLM + campos estruturados")),
            new Lab("04", "Decisões", "Próxima ação", "ABCD / atendimento",
                "Explore como um modelo pequeno escolhe o próximo passo de um atendimento.",
                "Escolha, probabilidades e indicação de revisão", "laboratorio-04-slm-decisoes-abcd.md", true,
                List.of("Histórico + regras", "Classificador calibrado", "Próxima ação + confiança")),
            new Lab("05", "Cálculos financeiros", "Destilação offline", "FinQA / relatórios",
                "Proponha cálculos verificáveis e compare um aluno destilado com seu professor.",
                "Programa, resultado executado e evidências", "laboratorio-05-slm-financeiro-finqa.md", false,
                List.of("Tabela + contexto + pergunta", "Professor 8B + validação", "Aluno 1.7B + cálculo verificável")));
    }
}
