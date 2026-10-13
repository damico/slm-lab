# Avaliador web dos laboratórios de SLMs

Aplicação Spring Boot + Thymeleaf Layout Dialect + Bootstrap 5, com comparação opcional por Spring AI. A interface usa as referências visuais fornecidas em `/tmp/design`: cartões escuros, números grandes, destaques roxos e verdes e layout responsivo. Os recursos de Bootstrap e Chart.js são WebJars locais; o navegador não consulta CDN.

O Lab 04 usa a inferência real do adaptador treinado, com probabilidades calibradas, catálogo de ações, exemplos ABCD e regras recuperadas. O Lab 02 usa o adaptador SQL treinado, os 11 bancos SQLite do Mini-Dev e as respostas preservadas do benchmark. Os Labs 01, 03 e 05 têm páginas de prévia até que seus adaptadores e APIs estejam prontos. Os números do dashboard são o benchmark registrado no roteiro, e não estatísticas dos novos testes da sessão.

## Iniciar na máquina com as GPUs

Requisitos: Java 17 ou superior, Maven e o venv compartilhado dos laboratórios. Se Java ou Maven estiverem ausentes no Ubuntu, o comando abaixo baixa o JDK e a ferramenta de compilação dos repositórios do sistema:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk maven
```

Se a API antiga já estiver rodando na porta 8084, encerre-a com Ctrl+C no terminal original antes de iniciar esta versão: o avaliador usa as novas rotas de catálogo, exemplos e contexto. No primeiro terminal, inicie a API. O script usa `~/lab04-decisoes/data`, `runs/main/final_adapter` e `runs/decision_policy.json`, e o Python de `~/slm-labs/.venv`. Ele executa o código do repositório diretamente, evitando depender das cópias antigas na pasta do lab. O modelo é carregado uma vez antes de a API começar a atender.

```bash
cd ~/git/slm-lab/avaliador
./start-lab04.sh
```

Espere a mensagem `API local em http://127.0.0.1:8084/decide`. Em outro terminal:

```bash
cd ~/git/slm-lab/avaliador
./run.sh
```

Abra **http://localhost:8080**. Em outra máquina da mesma rede, use o endereço da máquina que executa o Spring, por exemplo **http://192.168.68.137:8080**.

A primeira compilação baixa as dependências Java e os assets WebJars do Maven Central. As versões são fixadas no `pom.xml`; as próximas compilações reaproveitam o cache do Maven. `run.sh` grava o log da compilação e do Spring em `logs/web-DATA-HORA-PID.log`. A API Python grava seu próprio log em `logs/python-DATA-HORA-PID.log`.

## Iniciar também o Lab 02

Na máquina com os bancos e pesos, abra um terminal adicional:

```bash
cd ~/git/slm-lab/avaliador
./start-lab02.sh
```

Espere `API SQL pronta em http://127.0.0.1:8082`. O script usa `~/slm-labs/02-sql/runs/main_r16_lr1e4/final_adapter`, `data/db_manifest.json` e as predições em `results`, carrega o modelo uma vez e grava log separado em `logs/sql-DATA-HORA-PID.log`. A revisão do modelo vem do `revisions.json` original. Não baixa datasets nem treina novamente. Por padrão, SQL usa a GPU 1 e decisões a GPU 0, permitindo manter as duas APIs ativas.

Reinicie o Spring com `./run.sh` depois de atualizar o repositório e abra **http://localhost:8080/labs/02**. A tela permite selecionar os 500 exemplos por banco, editar pergunta e evidence, consultar esquema, gerar SQL, conferir as linhas retornadas e confrontar EX-set/EX-bag com a referência. A referência é associada apenas ao banco, pergunta e evidence originais e nunca é enviada ao modelo. O histórico do navegador e a exportação JSON preservam o teste.

A comparação registrada mostra as respostas original, ajustada e OpenAI salvas no teste concluído, sem novas chamadas pagas. **Comparar com OpenAI ao vivo** faz uma chamada opcional com as mesmas mensagens de sistema/usuário e reserva de 512 tokens. Tokens e custo estimado aparecem na tela; a chave segue somente no servidor Spring.

O executor reutiliza as restrições do `score_sql.py`: leitura, autorizador, extensões desativadas, timeout por progress handler de 15 segundos e limite de 100.000 linhas. A interface exibe até 100 linhas, indicando cortes. Somente bancos do Mini-Dev cadastrados no manifesto ficam disponíveis; o navegador não escolhe caminhos no filesystem. O botão de execução manual do SQL editado mostra uma nova consulta, sem reaproveitar a comparação do SQL anterior. Esse serviço é para o laboratório local; o limite SQLite não substitui isolamento de processo para um serviço público.

[Resultados medidos do Lab 02](docs/lab02-resultados-2026-10-12.json): especialista 25,50% EX-set, original 14,54%, OpenAI mini 51,45%, em 447 perguntas elegíveis com referências válidas. Os limites e denominadores completos estão no roteiro SQL.

## Alternar entre treinamento e demonstração

As APIs mantêm modelos carregados nas GPUs mesmo quando não recebem requisições. Antes de qualquer treino, encerre `start-lab02.sh` e `start-lab04.sh` com Ctrl+C nos seus terminais e confira os processos em `nvidia-smi`. Se foram iniciadas em segundo plano, identifique o PID de `serve_sql.py` ou `serve_decisions.py` e encerre somente esses processos com `kill -TERM PID`. Não encerre todos os processos Python indiscriminadamente.

SQL ocupa cerca de 2,6 GiB da GPU 1 e decisões cerca de 2 GiB da GPU 0 nesta máquina. Esse uso concorre com o treino e pode causar OOM nas RTX 3070 de 8 GB. O Spring pode continuar aberto; as APIs ficarão offline até você reiniciá-las depois do treino. O treino não precisa das APIs web.

## Ajustar caminhos e portas

| Variável | Padrão | Finalidade |
|---|---|---|
| `LAB02_DIR` | `~/slm-labs/02-sql` | Dados, bancos, resultados e revisão do Lab SQL |
| `LAB02_ADAPTER` | `LAB02_DIR/runs/main_r16_lr1e4/final_adapter` | Adaptador SQL treinado |
| `LAB02_GPU` | `1` | GPU física da API SQL |
| `LAB02_API_PORT` | `8082` | Porta Python SQL |
| `LAB02_API_URL` | `http://127.0.0.1:8082` | Conexão Spring com SQL |
| `LAB04_DIR` | `~/lab04-decisoes` | Pasta com dados, adaptador e calibração |
| `LAB_VENV_DIR` | `~/slm-labs/.venv` | Venv compartilhado já instalado |
| `LAB04_GPU` | `0` | GPU física usada pela API Python |
| `LAB04_API_PORT` | `8084` | Porta da API Python |
| `LAB04_API_URL` | `http://127.0.0.1:8084` | Endereço da API usado pelo Spring |
| `AVALIADOR_PORT` | `8080` | Porta da aplicação web |

Exemplo para usar a segunda GPU:

```bash
LAB04_GPU=1 ./start-lab04.sh
```

Se o Spring rodar em outro computador, a API Python pode continuar vinculada ao localhost da máquina de inferência. Abra um túnel SSH no computador do Spring:

```bash
ssh -L 8084:127.0.0.1:8084 -L 8082:127.0.0.1:8082 jdamico@192.168.68.137
```

Nesse caso, mantenha `LAB04_API_URL=http://127.0.0.1:8084`. Atualize também `LAB04_API_URL` se mudar a porta Python. Os nomes e a origem das variáveis são mostrados aqui; eles não alteram os caminhos do treino.

## Comparação opcional com OpenAI

A interface funciona sem chave de API. Para habilitar a comparação, defina a chave no terminal que inicia o Spring:

```bash
read -r -s -p 'Chave da API OpenAI: ' OPENAI_API_KEY
export OPENAI_API_KEY
export OPENAI_MODEL='gpt-4.1-mini'
./run.sh
```

Marque **Comparar com OpenAI** na tela de decisões. Uma execução faz uma chamada via Spring AI ChatClient, com `temperature=0`, `max_tokens=32` e o mesmo texto retornado por `/context`. Não adicionamos mensagens de sistema, gabaritos, ferramentas ou instruções de saída estruturada ao prompt do benchmark. O rótulo é validado contra o catálogo. Uma falha da API é mostrada ao lado do resultado local já obtido.

Essa opção envia o contexto para a OpenAI e tem custo. A chave fica no servidor e não é enviada ao navegador. Tokens e custo estimado aparecem no resultado; `OPENAI_INPUT_PRICE` e `OPENAI_OUTPUT_PRICE` configuram as tarifas por milhão (padrões 0,40 e 1,60 USD), consultadas em 12/10/2026. Atualize as tarifas ao trocar o modelo. `OPENAI_BASE_URL` permite configurar outro endereço compatível (padrão `https://api.openai.com`).

Referências: [Spring AI ChatClient](https://docs.spring.io/spring-ai/reference/api/chatclient.html), [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat), [gpt-4.1-mini e preços](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

## O que a tela faz

- Monta uma timeline editável de cliente, atendente e ações já executadas.
- Sorteia exemplos reais de `data/demo/case_*.json`; o gabarito é mostrado depois da inferência e fica fora da entrada enviada aos modelos.
- Atualiza os estágios por SSE: montagem do contexto, resultado do SLM e comparação opcional.
- Mostra escolha, confiança, limiar, revisão, 31 probabilidades, regras recuperadas, janela do histórico e contagem de tokens do Qwen.
- Acrescenta uma leitura simples da orientação ao atendente e, nos exemplos ABCD, as mensagens realmente entregues ao cliente e a última ação registrada.
- Mostra uma árvore interativa com histórico observado, etapas da inferência e continuação do ABCD. Cada nó abre seu conteúdo; zoom, arraste e expansão ajudam a acompanhar atendimentos longos.
- Permite importar `{"history":[...]}`, inspecionar o prompt exato e exportar o teste completo em JSON.
- Guarda até 20 testes neste navegador, com restauração do atendimento e do resultado salvo. Alterar o histórico remove o resultado anterior da tela.
- Alterna tema claro/escuro e se adapta a telas pequenas.

## Resposta entregue e evolução do exemplo

Use o seletor **Selecionar caso ABCD** para abrir um caso específico, como `case_01 / 4989:5`. Após a inferência, a seção **Resultado em linguagem simples** mostra a recomendação do SLM separada da resposta registrada no dataset. Nesse caso, o ABCD registra `search-timing` e informa ao cliente que o código promocional expira sete dias após sua emissão.

A seção **Evolução do atendimento** tem três ramos: eventos conhecidos pelo modelo, processamento/recomendação do teste e sequência humana registrada depois do ponto de teste. Os eventos posteriores e o gabarito permanecem fora do prompt do SLM e da comparação OpenAI. A continuação aparece depois do resultado; ela não afirma que o avaliador executou as ações. O diagrama é SVG local e interativo, sem dependência de CDN.

`start-lab04.sh` prepara automaticamente `data/demo_outcomes.json` a partir do ABCD já baixado quando esse arquivo ainda não existe. Se você usa uma pasta de fonte diferente, configure `LAB04_RAW_DIR`. Também pode preparar esse arquivo separadamente:

```bash
~/slm-labs/.venv/bin/python ~/git/slm-lab/scripts/lab04-decisoes/demo_outcomes.py \
  --data ~/lab04-decisoes/data --raw ~/lab04-decisoes/raw/abcd
```

A preparação normal de dados também passa a gerar esses registros. O arquivo liga cada exemplo à sua continuação e confere o histórico por hash; editar o atendimento elimina a associação com o desfecho original. Nos históricos livres, a tela apresenta a recomendação e os eventos observados, e informa que não há resposta posterior registrada. Testes antigos salvos no navegador podem ser associados ao registro atual quando o histórico coincide com um exemplo preparado.

A indicação de revisão e as probabilidades pertencem ao classificador calibrado. A referência generativa OpenAI retorna apenas um rótulo neste comparativo. `retrieve_utterance` indica conversar/coletar informação; a recomendação não executa ações.

## Contratos HTTP

| Serviço | Rota | Uso |
|---|---|---|
| Spring | `GET /api/lab02/status` | Conexão da API SQL |
| Spring | `GET /api/lab02/catalog` | Bancos, esquemas, exemplos e métricas SQL |
| Spring | `POST /api/lab02/predict/stream` | Geração e execução SQL com SSE |
| Spring | `POST /api/lab02/predict` | Mesmo resultado SQL em JSON |
| Spring | `POST /api/lab02/execute` | Executa SQL somente leitura no banco cadastrado |
| Spring | `GET /api/status` | Conexão da inferência e disponibilidade da comparação |
| Spring | `GET /api/lab04/catalog` | Classes e ações do adaptador |
| Spring | `GET /api/lab04/examples` | Exemplos reais preparados no lab |
| Spring | `POST /api/lab04/predict` | Resultado JSON completo |
| Spring | `POST /api/lab04/predict/stream` | Eventos SSE com estágios e resultado |
| Python | `GET /health`, `/catalog`, `/examples` | Metadados e exemplos |
| Python | `POST /context` | Prompt e regras produzidos pelo ContextBuilder |
| Python | `POST /decide` | Contrato existente de inferência, com tokens e latência adicionais |

Spring recebe `history` e `compare`; envia somente `history` ao Python. Os eventos aceitam `role`, `text` e `action`. Papéis válidos: `customer`, `agent`, `action`. O nome de uma ação deve ser do catálogo e aparecer num evento com papel `action`.

## Docker opcional para a camada web

O Compose usa rede do host no Linux para acessar a API Python local. Mantenha a inferência no venv que já funciona com as GPUs. O build Docker baixa as imagens oficiais de Maven/Java e as bibliotecas do projeto para empacotar o Spring; não baixa outro modelo nem copia os pesos.

```bash
cd ~/git/slm-lab/avaliador
docker compose up --build
```

## Verificação

Os testes Java cobrem renderização das telas/assets locais, validação da entrada, preservação do prompt, isolamento do gabarito e fluxo SSE. Os testes Python cobrem as novas rotas HTTP e o contrato do motor com um modelo substituto. Esses testes não executam inferência GPU nem chamadas pagas. O fluxo do navegador também foi verificado com o adaptador real da máquina de testes: decisão, 31 probabilidades, regras, exemplos e histórico de sessão. A comparação paga foi conferida com um endpoint substituto que valida o corpo enviado por Spring AI.

```bash
mvn test
python3 -m unittest discover -s tests -v
```

Para conferir a inferência real, inicie `start-lab04.sh` e execute um caso ABCD pela tela. Modelos, dados e pesos permanecem nas pastas do laboratório.

## Capturas da interface

[Dashboard](docs/dashboard.png), [Lab 04 com inferência real](docs/lab04.png), [resposta entregue](docs/lab04-desfecho.png) e [evolução do atendimento](docs/lab04-evolucao.png) e [Lab 02 com SQL e tabela reais](docs/lab02.png).

## Lab 05: FinQA e destilação offline

O dashboard inclui uma prévia do [quinto laboratório](../laboratorio-05-slm-financeiro-finqa.md), ainda sem adaptador treinado/inferência web. O roteiro oferece professor Qwen3-8B, aluno Qwen3-1.7B, filtragem da supervisão, dois treinos com os mesmos IDs, quatro condições avaliadas e API Python após o treino. A prévia não gera respostas simuladas.
