# Diagramas dos laboratórios

Os laboratórios 01 a 04 têm um fonte Mermaid (`.mmd`) e a imagem renderizada (`.png`); o Lab 05 disponibiliza seu fonte Mermaid. Os diagramas mostram as fases do roteiro: preparação, treino QLoRA, inferência/comparação e avaliação. Os laboratórios 03 e 04 incluem também a aplicação local e, no 04, a calibração.

| Laboratório | Fonte | Imagem |
|---|---|---|
| 01 - Ferramentas (xLAM) | [laboratorio-01-ferramentas.mmd](laboratorio-01-ferramentas.mmd) | [PNG](laboratorio-01-ferramentas.png) |
| 02 - SQL (BIRD) | [laboratorio-02-sql.mmd](laboratorio-02-sql.mmd) | [PNG](laboratorio-02-sql.png) |
| 03 - Recibos (SROIE) | [laboratorio-03-recibos.mmd](laboratorio-03-recibos.mmd) | [PNG](laboratorio-03-recibos.png) |
| 04 - Decisões (ABCD) | [laboratorio-04-decisoes.mmd](laboratorio-04-decisoes.mmd) | [PNG](laboratorio-04-decisoes.png) |

Edite o `.mmd` e gere a imagem novamente com o Mermaid CLI. Os diagramas 03 e 04 são mais largos e usam `-w 2000`:

```bash
npx -y @mermaid-js/mermaid-cli@11.4.2 -i laboratorio-01-ferramentas.mmd -o laboratorio-01-ferramentas.png -b white -s 2
npx -y @mermaid-js/mermaid-cli@11.4.2 -i laboratorio-03-recibos.mmd -o laboratorio-03-recibos.png -b white -w 2000 -s 2
```

No Ubuntu 23.10 ou posterior, o Chromium usado pelo CLI pode falhar com "No usable sandbox". Nesse caso, passe `-p puppeteer.json` com o conteúdo `{"args":["--no-sandbox"]}`.

O Lab 05 tem o fonte [laboratorio-05-finqa.mmd](laboratorio-05-finqa.mmd), com professor offline, validação, pares SFT/destilados e avaliação comum. A imagem pode ser renderizada com o mesmo Mermaid CLI quando necessário.
