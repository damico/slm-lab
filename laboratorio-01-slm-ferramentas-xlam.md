# Laboratório 1, SLM especialista em chamadas de ferramentas

**José Damico · Laboratório local · Versão 1.0 · 10/10/2026**

**Objetivo:** ajustar um Qwen3-1.7B com exemplos prontos de chamadas de ferramentas, avaliar o ganho em relação ao modelo original e comparar com um LLM maior. A preparação e o treinamento podem levar horas ou dias; a apresentação dos resultados cabe em uma hora.

## 1. Resultado que você construirá

Um componente que recebe uma solicitação e um catálogo de funções, e retorna um array JSON com os nomes e argumentos das chamadas. É aplicável à seleção de ferramentas em agentes do InfiniteStack: a aplicação valida e executa a chamada. O SLM não precisa armazenar resultados de consultas nem implementar o protocolo MCP.

O dataset é [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k): 60 mil exemplos sintéticos de solicitações, ferramentas e respostas, construídos pelo pipeline APIGen, com validações de formato, execução e semântica. O catálogo de origem abrange 3.673 APIs em 21 categorias. Há dados em inglês e condições de acesso a aceitar no Hugging Face. Consulte a licença e os termos na página antes de redistribuir os dados.

A hipótese é **especialização funcional mensurável**. O artigo APIGen de 2024 reportou um modelo de aproximadamente 1B superando GPT-3.5-Turbo e Claude-3 Haiku em seu benchmark da época. Isso é uma referência histórica; este laboratório usa outro modelo/receita e precisa produzir suas próprias medidas contra um LLM atual.

Escopo inicial: uma solicitação por exemplo, seleção de uma ou mais chamadas independentes e preenchimento de argumentos. Não demonstrará, sozinho, planejamento de vários passos, diálogo longo, competência agronômica ou habilidade geral de um agente.

Crie uma pasta própria:

```bash
mkdir -p "$HOME/slm-labs/01-tools"
cd "$HOME/slm-labs/01-tools"
```

Use este roteiro junto dos scripts no repositório. O laboratório SQL possui sua própria pasta de scripts e pode ser executado separadamente.

## 2. Ambiente: Ubuntu e duas RTX 3070 de 8 GB

Este roteiro considera **Linux/Ubuntu, Bash, Python 3.11 ou 3.12, duas placas de 8 GB e aproximadamente 32 GB de RAM**. Se o computador estiver em Windows, execute em Ubuntu/WSL2 com CUDA funcionando, ou use Ubuntu nativo; os comandos abaixo não são de PowerShell.

Use o ambiente compartilhado descrito em [Preparação geral](preparacao-geral.md). Execute os dois laboratórios em sequência quando cada um utilizar as duas GPUs. Feche outras inferências que ocupem VRAM. Reserve inicialmente 25–40 GB de disco para ambiente, cache do modelo e checkpoints; no laboratório BIRD, acrescente espaço para os bancos e os ZIPs, conforme a seção de download.

**As memórias das placas não se somam automaticamente.** O treinamento usa Distributed Data Parallel (DDP): um processo e uma cópia quantizada do modelo em cada GPU. Cada exemplo precisa caber nos 8 GB de uma placa. A segunda GPU distribui exemplos e sincroniza os gradientes dos adaptadores. O ganho de velocidade precisa ser medido; não se presume 2×.

Verifique o driver existente:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
python3 --version
```

Se `nvidia-smi` funciona, não há razão para reinstalar o driver para começar. A indicação “CUDA Version” desse comando descreve a capacidade do driver; o PyTorch instala seu runtime CUDA. A preparação geral usa a distribuição oficial cu128 do PyTorch 2.7.1, que requer um driver compatível. Se seu driver não for compatível, consulte a matriz oficial indicada nas referências antes de trocar a distribuição.

Prepare primeiro o ambiente compartilhado conforme [Preparação geral](preparacao-geral.md). Dentro da pasta do laboratório criada na seção 1, ative esse ambiente:

O `cp` abaixo copia a receita única de dependências diretamente do repositório para a pasta do experimento. `python -m pip freeze` registra as versões efetivamente instaladas no venv ativo. Esses comandos não dependem de arquivos de requisitos em `~/slm-labs`, não criam outro venv e não instalam pacotes. Ajuste o caminho do repositório se necessário.

```bash
source ~/slm-labs/.venv/bin/activate
cp ~/git/slm-lab/requirements.txt ./requirements.txt
python -m pip freeze > requirements-resolved.txt
mkdir -p data runs logs results raw
export TOKENIZERS_PARALLELISM=false
```

As versões foram fixadas para tornar a receita explícita, não por serem as versões mais recentes. O arquivo `requirements-resolved.txt` registra também as dependências transitivas realmente instaladas. A receita precisa passar pelo teste curto no seu equipamento antes de um treino longo.

```bash
python ~/git/slm-lab/verificar_ambiente.py
```

O script escolhe BF16 quando o runtime informa suporte; caso contrário, usa FP16. Não usa FlashAttention como dependência adicional. `attn_implementation="sdpa"` deixa o PyTorch selecionar um caminho de atenção compatível.

## 3. Como executar os scripts deste documento

Os scripts estão em [scripts/lab01-ferramentas](scripts/lab01-ferramentas) no repositório. Dentro da pasta de trabalho do laboratório, copie-os antes de seguir:

`cp -n` copia os scripts da pasta específica deste laboratório no repositório para a pasta atual. Isso permite executar os comandos `python arquivo.py` e resolver os imports locais, mantendo dados e resultados isolados dos outros laboratórios. A opção `-n` preserva arquivos de destino que já existem.

```bash
cp -n ~/git/slm-lab/scripts/lab01-ferramentas/*.py ./
```

O exemplo considera o repositório em `~/git/slm-lab`; ajuste o caminho se necessário. `cp -n` preserva scripts existentes. Para atualizar uma cópia antiga, compare-a com a versão do repositório antes de substituí-la.

O `lab_runtime.py` contém carregamento, treinamento e inferência. O `metrics.py` calcula métricas, intervalos de Wilson e diferenças pareadas. Os demais scripts são específicos do laboratório.

O modelo inicial é **Qwen/Qwen3-1.7B**, em modo sem raciocínio explícito (`enable_thinking=False`). O treino aplica a perda somente à resposta; tokens do prompt e padding recebem rótulo `-100`. A base fica quantizada em NF4 e congelada; os adaptadores LoRA são treinados nas camadas lineares. Isso é fine-tuning supervisionado com QLoRA, e não pré-treinamento do zero.

Na primeira preparação, os scripts registram os commits do modelo e datasets em `revisions.json`. Reexecuções reutilizam essas revisões. Preserve esse arquivo com os resultados.

## 4. Preparar o dataset pronto

1. Abra a página do xLAM, entre na sua conta e aceite as condições de acesso.
2. No terminal do ambiente virtual, autentique-se. Não coloque o token diretamente nos comandos salvos.

O comando `prepare_tools.py` baixa do Hugging Face os exemplos do xLAM: solicitações, catálogos de ferramentas e chamadas corretas. Esses dados fornecem os pares de entrada e saída para especializar o SLM e comparar suas previsões com o gabarito. A autenticação permite acessar os arquivos conforme as condições do dataset. O script também baixa o tokenizer do Qwen3-1.7B para formatar os prompts e medir seus comprimentos antes de gerar os JSONLs.

```bash
hf auth login
python prepare_tools.py --out data --max-seq 2048 --max-new 512
cat data/source_report.json
cat data/length_report.json
```

A divisão é determinística por **catálogo completo de ferramentas**: aproximadamente 80% dos grupos para treino, 10% para validação e 10% para teste. As proporções por linha podem diferir. Consultas textualmente duplicadas são removidas. Um mesmo catálogo não aparece em splits diferentes, mas uma API individual pode aparecer em catálogos distintos. Portanto, este teste mede novas solicitações/combinações de catálogo, **não garante APIs totalmente inéditas**. Para medir isso, crie posteriormente uma separação por famílias de APIs e reporte-a como outro experimento.

O script só mantém os campos necessários no prompt e na saída. Não inclui a resposta correta no prompt. `answer`/`gold` servem ao treinamento/avaliador; a inferência transmite somente `messages`.

Inspecione um exemplo e confirme que o alvo corresponde às funções disponíveis:

```bash
python inspecionar_dados.py
```

Para um piloto de engenharia, `--train-limit 2000` reduz o treino; não o confunda com o experimento completo. Use uma pasta distinta, por exemplo `--out data_pilot`, e passe `--data data_pilot` ao treino. O teste e a validação continuam independentes, mas não olhe o teste para escolher o tamanho do piloto.

## 5. Arquivos Python necessários

Os arquivos principais estão listados abaixo. Copie os scripts conforme a seção 3 antes de executar a preparação da seção 4.

### lab_runtime.py

Código: [lab_runtime.py](scripts/lab01-ferramentas/lab_runtime.py).

### metrics.py

Código: [metrics.py](scripts/lab01-ferramentas/metrics.py).

### prepare_tools.py

Código: [prepare_tools.py](scripts/lab01-ferramentas/prepare_tools.py).

### score_tools.py

Código: [score_tools.py](scripts/lab01-ferramentas/score_tools.py).

## 6. Verificar memória e treinar

Antes de treinar, encerre as APIs de inferência dos Labs 02 e 04 nos terminais em que foram iniciadas. Elas mantêm os pesos na VRAM mesmo sem requisições. Treinar com duas GPUs requer memória livre em ambas; uma API ocupando cerca de 2,6 GiB pode provocar OOM já nos primeiros passos. Confira os processos com `nvidia-smi`. A interface Spring pode permanecer aberta, mas as APIs ficam offline durante o treino. Depois, reinicie-as com os scripts em `avaliador/`.

O prompt `(.venv)` não identifica qual ambiente está ativo. Os blocos abaixo ativam explicitamente `~/slm-labs/.venv`, criam `logs` e usam `pipefail`, para que uma falha no treino ou no `tee` não seja mascarada.

Nos comandos `torchrun`, o separador `--` antes do nome do script encerra as opções do lançador. Assim, `--run` e os demais argumentos seguintes chegam ao script de treino sem serem interpretados como abreviações de opções do `torchrun`.

Na primeira execução do treino ou da inferência local, o runtime baixa do Hugging Face os pesos e a configuração do Qwen3-1.7B que ainda não estiverem no cache. Esses arquivos são a base pré-treinada sobre a qual o laboratório ajusta os adaptadores e, no laboratório 04, a camada de classificação. Inferências posteriores reutilizam o cache e carregam o adaptador da pasta local indicada; se algum arquivo do tokenizer ou da base estiver ausente, ele também é obtido do Hugging Face.

A configuração inicial é: **contexto total 2.048 tokens, reserva de até 512 tokens para saída, LoRA rank 16, learning rate 0,0001, microbatch 1, acumulação 8 e duas épocas**. É um ponto de partida, não uma garantia de encaixe em 8 GB nem uma configuração ótima demonstrada. O contexto inclui o catálogo/esquema, a pergunta e a resposta.

Primeiro, faça um teste de 10 passos em uma GPU. Ajuste o índice se a GPU 1 não estiver livre:

```bash
source ~/slm-labs/.venv/bin/activate
mkdir -p logs runs
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py train \
  --data data --run runs/smoke_1gpu \
  --max-steps 10 --accum 1 --eval-limit 32 \
  2>&1 | tee logs/smoke_1gpu.log
```

Depois, verifique o caminho de duas GPUs:

```bash
source ~/slm-labs/.venv/bin/activate
mkdir -p logs runs
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -- \
  lab_runtime.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-limit 32 \
  2>&1 | tee logs/smoke_2gpu.log
```

Os adaptadores desses testes curtos são apenas diagnósticos. O treinamento principal começa novamente do modelo original em outra pasta:

```bash
source ~/slm-labs/.venv/bin/activate
mkdir -p logs runs
set -o pipefail
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_train.csv &
LAB_MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  torchrun --standalone --nproc-per-node=2 -- lab_runtime.py train \
  --data data --run runs/main_r16_lr1e4 \
  --epochs 2 --accum 8 --rank 16 --lr 0.0001 --eval-limit 256 \
  2>&1 | tee logs/main_train.log
kill "$LAB_MONITOR_PID"
```

O batch efetivo é `1 × 8 × 2 = 16` exemplos por atualização. Se treinar em uma GPU e quiser preservar esse batch, use `--accum 16`. Isso não torna o tempo de execução idêntico.

A cada 100 passos, o Trainer avalia a perda em até 256 exemplos fixos da validação e salva um checkpoint. Ao terminar, carrega o melhor checkpoint observado por `eval_loss` e salva `runs/main_r16_lr1e4/final_adapter`. Em testes com menos de 100 passos pode não haver avaliação; nesse caso, o adaptador final é apenas o último estado. A perda ajuda a selecionar checkpoints, mas a avaliação funcional da seção 7 é a medida importante.

Não existe previsão confiável de duração sem executar o teste curto. Use os segundos por passo do seu log e estime o total de passos com `ceil(n_treino / 16) × épocas`; some avaliação e gravação. Não extrapole apenas do primeiro passo, que inclui aquecimento.

Para retomar um treino interrompido, use uma pasta de checkpoint realmente existente:

```bash
python listar_checkpoints.py
# Exemplo de forma, substituindo pelo caminho que o comando acima mostrou:
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -- lab_runtime.py train \
#   --data data --run runs/main_r16_lr1e4 --epochs 2 --accum 8 \
#   --resume runs/main_r16_lr1e4/checkpoint-N
```

Preserve dados, revisões e hiperparâmetros ao retomar. O arquivo `run_report.json` registra configuração, hashes dos dados, tempo e pico de memória alocada no processo da GPU de rank 0. `gpu_train.csv` registra o uso observado das duas placas, incluindo consumo de outros processos.

## 7. Comparar o modelo original, o ajustado e um LLM maior

Use primeiro a **validação** para ajustar prompt, taxa de aprendizado, número de épocas e decodificação. Faça poucas experiências deliberadas: por exemplo, uma e duas épocas, ou learning rates `0.00005` e `0.0001`. Troque o nome de `--run` a cada treino. Não selecione a melhor configuração olhando o teste final.

Para uma verificação funcional rápida, crie um conjunto de até 50 casos elegíveis da validação:

```bash
python preparar_demo.py
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/validation_demo.jsonl --output results/val_base.jsonl
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/validation_demo.jsonl --output results/val_tuned.jsonl \
  --adapter runs/main_r16_lr1e4/final_adapter
```

Avalie esses arquivos com o avaliador da seção 8, apontando a referência para `data/validation_demo.jsonl`. Depois de escolher sua configuração, execute o **teste completo**, uma vez por condição:

```bash
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_base.jsonl
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_tuned.jsonl \
  --adapter runs/main_r16_lr1e4/final_adapter
```

Os dois modelos locais usam a mesma revisão da base, quantização, prompt, GPU e limite de saída. O padrão do tutorial é decodificação gulosa para uma primeira comparação determinística, sem modo thinking. A documentação Qwen recomenda amostragem no uso geral: `--sample` usa temperatura 0,7, top-p 0,8 e top-k 20. Se escolher amostragem na validação, aplique-a a ambas as condições e repita com sementes fixas, por exemplo 42, 43 e 44, em arquivos distintos. Não escolha a melhor resposta entre tentativas sem declarar um experimento diferente.

Para a referência por API, use **OpenAI `gpt-4.1-mini`**, como nos Labs 02 e 04. O runtime já aceita o endpoint oficial de Chat Completions e registra modelo, parâmetros, tokens e latência. A chave é lida sem eco e não deve ser salva no Markdown nem no Git. Configure e execute os comandos no mesmo terminal:

```bash
export LLM_URL='https://api.openai.com/v1/chat/completions'
export LLM_MODEL='gpt-4.1-mini'
read -r -s -p 'Chave da API OpenAI: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"max_tokens":512}'
```

`max_tokens=512` limita a resposta com o array JSON de nomes e argumentos. Mantenha a mesma reserva de saída usada na preparação e nas inferências locais. O catálogo de ferramentas continua dentro do prompt; esta condição não ativa o tool calling nativo da OpenAI. Nenhuma ferramenta é executada: o experimento compara as chamadas propostas. Se houver `output_limit`, a saída truncada conta como falha; ajuste o orçamento na validação e reavalie todas as condições antes de congelar o teste.

O `cp -n` abaixo copia somente o novo contador do repositório para a pasta `01-tools`, sem substituir um arquivo existente. Ele permite executar o helper ao lado das cópias locais do runtime; ajuste o caminho do repositório se necessário.

Antes de chamar a API, conte os casos elegíveis. O número depende da divisão por catálogo e dos limites de contexto: não assuma que o teste contém exatamente 10% das 60 mil linhas. Depois, faça uma chamada de validação para conferir compatibilidade e consumo:

```bash
cp -n ~/git/slm-lab/scripts/lab01-ferramentas/count_api_calls.py ./
python count_api_calls.py --test
python lab_runtime.py predict --api --limit 1 \
  --input data/validation_demo.jsonl --output results/api_smoke.jsonl
python count_api_calls.py --predictions results/api_smoke.jsonl
```

Esse arquivo de uma chamada serve apenas para inspeção. Não o pontue contra os 50 casos como se fosse uma avaliação completa. Se houver `api_error`, corrija credenciais, endpoint ou parâmetros antes de continuar. Então:

```bash
python lab_runtime.py predict --api \
  --input data/test.jsonl --output results/test_llm.jsonl
python count_api_calls.py --predictions results/test_llm.jsonl
unset LLM_API_KEY
```

O contador informa **N chamadas previstas**, uma por exemplo elegível, mais a chamada do piloto. Os exemplos `context_exceeded` não chamam a API. Erros transitórios podem causar até três tentativas por exemplo no runtime; portanto, N descreve as chamadas lógicas, não um limite de requisições HTTP. Não é necessário usar uma GPU para este caminho.

**Estimativa de custo:** tarifas padrão do `gpt-4.1-mini` verificadas em 13/10/2026: US$ 0,40 por milhão de tokens de entrada e US$ 1,60 por milhão de saída. Consulte os [preços oficiais](https://developers.openai.com/api/docs/models/gpt-4.1-mini) ao executar. Para N chamadas:

`custo_USD = N × (tokens_médios_entrada × 0,40 + tokens_médios_saída × 1,60) / 1.000.000`

| Cenário ilustrativo | Entrada / saída por chamada | 1.000 chamadas | 5.000 chamadas |
|---|---|---:|---:|
| Resposta JSON curta | 1.000 / 128 tokens | US$ 0,605 | US$ 3,024 |
| Catálogo maior e saída no limite | 2.048 / 512 tokens | US$ 1,638 | US$ 8,192 |

Esses cenários são premissas, não medições nem tetos garantidos: o tokenizer da OpenAI difere do Qwen, e o tamanho do catálogo varia. O contador soma o `usage` retornado pela API e projeta o custo do teste a partir da média do arquivo informado. A estimativa usa entrada sem desconto de cache e não substitui o faturamento do provedor. Um piloto de uma chamada pode subestimar ou superestimar o conjunto; use mais casos de validação em um arquivo novo se precisar de uma previsão melhor.

**Estimativa de tempo:** o runtime faz chamadas sequenciais. Use `tempo ≈ N × latência_média`; com 1 segundo por chamada, 1.000 chamadas levariam cerca de 16min40s e 5.000 cerca de 1h23min20s. Isso é um cenário, não uma previsão medida para xLAM. O contador mostra o ritmo observado e a projeção baseada nas respostas `ok` do piloto; erros, retentativas, limites de taxa e variações de tamanho podem aumentar o tempo.

**Execução concluída em 14/10/2026:** 5.996 linhas registradas e elegíveis, com 5.992 respostas `ok`, três `api_error` (`URLError`) e uma `output_limit`. As quatro falhas permanecem nas métricas. As 5.993 respostas com `usage` somaram **2.010.445 tokens de entrada e 249.251 de saída**. Pelas tarifas configuradas de US$ 0,40/US$ 1,60 por milhão, o consumo registrado custa **US$ 1,202980**, aproximadamente **US$ 0,201 por 1.000 exemplos**. Isso exclui o smoke test e consumo não informado em retentativas; não é uma conferência da fatura.

A latência média das respostas `ok` foi **1,033 s**, equivalente a **58,1 chamadas/min**. Os **103,2 minutos** do contador são uma projeção sequencial baseada nessa média, não uma duração total cronometrada: falhas, retentativas e preparação podem aumentar o tempo.

`--limit` permite um piloto, mas a comparação final deve usar os mesmos IDs em todas as condições. Para repetir uma execução, use outro arquivo de saída: o script impede sobrescrita acidental e não retoma uma inferência parcial automaticamente. Depois, pontue `results/test_llm.jsonl` com `score_tools.py` e compare com os modelos locais na seção 8.

**Protocolo de comparação:** todos recebem a mesma informação no prompt e produzem uma resposta textual no contrato especificado. Este script não ativa o modo nativo de tool calling ou saídas estruturadas do provedor. Um LLM com suporte nativo pode obter outro resultado; essa configuração merece uma linha adicional na comparação de produção, com seu próprio cliente e adaptação de formato. Não apresente a condição de prompt uniforme como o melhor desempenho possível de cada plataforma.

A preparação marca `eligible=false` quando o prompt mais a reserva de saída excede o contexto escolhido. Esses casos continuam no teste, recebem `context_exceeded` em todas as condições e aparecem na cobertura. Assim, não se corta silenciosamente um esquema ou catálogo. A elegibilidade do teste não usa o tamanho da resposta correta. Os dados de treino/validação também excluem respostas que excedam o contexto; as contagens ficam nos relatórios.

## 8. Avaliar chamadas e argumentos

```bash
python score_tools.py data/test.jsonl results/test_base.jsonl --out results/base
python score_tools.py data/test.jsonl results/test_tuned.jsonl --out results/tuned
python score_tools.py data/test.jsonl results/test_llm.jsonl --out results/llm
python metrics.py results/base.cases.jsonl results/tuned.cases.jsonl
python metrics.py results/llm.cases.jsonl results/tuned.cases.jsonl
```

Para a validação rápida, por exemplo:

```bash
python score_tools.py data/validation_demo.jsonl results/val_base.jsonl --out results/val_base
python score_tools.py data/validation_demo.jsonl results/val_tuned.jsonl --out results/val_tuned
```

A métrica principal exige **todas as chamadas e todos os argumentos corretos**. Objetos JSON são normalizados; ordem das chaves é ignorada. A ordem entre chamadas independentes também é ignorada, mas sua multiplicidade é preservada. Ordem dentro de listas de argumentos permanece significativa. O avaliador é estrito quanto a valores e representação numérica serializada: casos como `1` e `1.0` podem ser julgados diferentes. Se normalizar tipos com base no schema, faça-o igualmente para todas as condições e registre a mudança.

Também são calculados:

- `schema_valid_rate`: saída é um array de objetos `name`/`arguments`; isso verifica o envelope, não toda a semântica do schema de cada API.
- `tool_names_correct_rate`: nomes e quantidade de chamadas corretos, mesmo se os argumentos estiverem errados.
- `unknown_tool_rate`: proporção de respostas que inventam uma ferramenta fora do catálogo fornecido.

O avaliador aceita um único bloco Markdown envolvendo a resposta, mas não procura seletivamente JSON no meio de uma explicação. Chaves JSON duplicadas, constantes inválidas, texto extra e chamadas incompletas são rejeitados.

**As funções do dataset não são executadas por este avaliador.** Ele compara contra o gabarito; validação de execução feita pelos autores não significa execução no seu laboratório. Para uma demonstração operacional, escolha uma ferramenta sua de leitura, valide nome e tipos no backend e execute somente após essa validação. Apresente o resultado como teste adicional de integração, separado da acurácia do dataset.

Uma extensão útil é avaliar categorias single-turn do [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard), usando seu avaliador e adaptador de modelo. Esse passo não está automatizado aqui e não é necessário para montar o primeiro laboratório. **Os números produzidos por `score_tools.py` são do holdout xLAM local; não são pontuações BFCL.** Não compare diretamente com o leaderboard.

O dataset deste roteiro não assegura cobertura adequada de solicitações impossíveis ou parâmetros ausentes. A capacidade de abstenção precisa de exemplos e avaliação específicos; não conclua que o ajuste aprendeu a recusar só porque gerou JSON correto nos casos cobertos.

## 9. Interpretar os resultados e demonstrar valor

Cada avaliação produz `*.summary.json` e `*.cases.jsonl`. Os [resumos originais preservados](avaliador/docs/lab01-resultados-2026-10-14.json) permitem conferir os números. Resultados medidos em 14/10/2026, com 5.996 referências válidas e elegíveis nas três condições:

| Condição | Acertos completos | Acurácia | IC 95% Wilson | Latência p50 / p95 | Pico alocado na inferência |
|---|---:|---:|---|---|---|
| Qwen3-1.7B original | 3.771 / 5.996 | 62,89% | 61,66–64,11% | 3,79 / 7,76 s | 1,497 GiB |
| Qwen3-1.7B + QLoRA | 4.875 / 5.996 | **81,30%** | 80,30–82,27% | 3,77 / 7,79 s | 1,615 GiB |
| OpenAI `gpt-4.1-mini` | 4.516 / 5.996 | 75,32% | 74,21–76,39% | 0,92 / 1,51 s | Não observável em API |

A diferença pareada especialista menos base foi **+18,41 pp**, com bootstrap 95% de **+17,26 a +19,55 pp**; contra a API, **+5,99 pp**, com intervalo de **+5,02 a +6,97 pp**. O bootstrap atual reamostra exemplos, não grupos de catálogo; ele não ajusta uma eventual correlação entre exemplos do mesmo catálogo. A comparação usa o contrato textual e o mesmo prompt, sem tool calling nativo da API. Não é uma pontuação BFCL nem uma alegação de superioridade geral sobre o LLM.

| Medida auxiliar | Original | Especialista | API |
|---|---:|---:|---:|
| Contrato JSON válido | 97,55% | 99,82% | 99,85% |
| Conjunto de nomes correto | 93,05% | 98,78% | 98,75% |
| Ferramenta fora do catálogo | 0,083% | 0% | 0% |
| Respostas usadas na latência | 5.989 | 5.991 | 5.992 |

O ajuste manteve uma latência próxima da base; a API teve menor latência neste experimento. Seu custo registrado foi aproximadamente US$ 0,201 por 1.000 exemplos. O custo local não foi estimado: exige consumo elétrico e amortização do equipamento.

O treino completo usou **46.364 exemplos**, duas épocas, QLoRA rank 16, LR 0,0001, acumulação 8 e duas GPUs. O `Trainer` registrou **37.858,85 s (10h30min59s)**; o runtime registrou **37.903,34 s (10h31min43s)**, incluindo a finalização. O pico alocado no rank 0 durante o treino foi **6,292 GiB**. Esses picos são medições do allocator PyTorch, não o uso total de memória mostrado pelo `nvidia-smi`.

A validação demonstrativa de 50 casos teve **35 acertos (70%)** na base e **43 (86%)** no especialista. Sua latência p50/p95 foi 3,55/5,90 s e 3,65/6,24 s, respectivamente. Essa amostra não substitui o teste completo. Os metadados das três inferências confirmam o mesmo hash de entrada `95afb4d78a8e641a7f33e2a602404e8d25639d48d3f2ffc568bb35b508014015`, decodificação gulosa local, sem thinking e reserva de 512 tokens.

### Experimentar no avaliador web

O [avaliador](avaliador/README.md) oferece a página `/labs/01`, com solicitação, catálogo, chamadas propostas, comparação registrada e diagrama das etapas. `avaliador/start-lab01.sh` inicia por padrão o benchmark em CPU: as respostas já geradas podem ser exploradas enquanto outro laboratório ocupa as GPUs. Alterar o pedido ou catálogo remove a associação com o gabarito; uma entrada nova requer o modo de inferência real.

Depois de liberar uma GPU, inicie o mesmo serviço com `LAB01_ENABLE_GPU=1 LAB01_GPU=0 bash avaliador/start-lab01.sh`. Ele carrega o adaptador de `runs/main_r16_lr1e4/final_adapter` somente na primeira inferência e o reutiliza. A primeira carga pode baixar pesos ou tokenizer ausentes do cache; eles fornecem a base pré-treinada usada pelo adaptador. O frontend distingue uma nova geração de uma resposta registrada. As ferramentas são propostas, sem execução externa. A comparação OpenAI ao vivo é opcional e utiliza Spring AI com o mesmo prompt, sem enviar o gabarito.

`accuracy_eligible` usa apenas casos elegíveis e com referência válida; `accuracy_all_valid` também penaliza os casos que não couberam no contexto. Mostre ambos e a cobertura. Predições ausentes, erros de API e respostas que atingiram o limite de saída contam como falhas, não desaparecem da acurácia. Latências p50/p95 são calculadas sobre requisições bem-sucedidas; o número desse subconjunto aparece no relatório.

Os tempos locais excluem carregamento do modelo e incluem geração após um aquecimento. O tempo de API inclui rede, serviço e eventuais tentativas. Isso mede a experiência desses dois caminhos, não uma comparação isolada da velocidade dos chips. Para latência mais estável, faça três passagens separadas, sem outro processo utilizando as GPUs, e reporte a dispersão. Não compare tokens/segundo de tokenizadores diferentes como se fossem uma unidade linguística idêntica.

Os scripts registram o consumo de tokens retornado pela API. Com preços vigentes por milhão de tokens de entrada e saída, calcule:

`custo_API = tokens_entrada / 1e6 × preço_entrada + tokens_saída / 1e6 × preço_saída`

Depois normalize pelo número de solicitações. Inclua tarifas específicas de cache/raciocínio quando existirem. Custo local não é zero: declare energia, amortização do equipamento, tempo de operação e quantas inferências amortizam o treinamento. `power.draw` mede as GPUs, não o computador inteiro; não o apresente como consumo total da máquina.

O intervalo de Wilson fornecido é uma aproximação binomial por exemplo. Para diferença entre modelos, use a comparação pareada da seção 8. O bootstrap também é por exemplo; correlações dentro do mesmo catálogo ou banco podem estreitar artificialmente o intervalo. Para publicação científica, adicione bootstrap por grupo/banco e um teste independente mais amplo.

Resultados possíveis que justificam uma boa aula:

1. O ajuste melhora claramente o pequeno modelo e aproxima sua qualidade do LLM maior.
2. O especialista supera o LLM maior **no recorte avaliado**, com diferença sustentada pelos dados.
3. O especialista mantém qualidade suficiente para um requisito definido previamente, com uma vantagem operacional medida.
4. O LLM maior continua melhor, e o laboratório mostra onde a especialização não compensou.

Não escolha somente exemplos em que o SLM vence. Selecione para a demonstração: um acerto compartilhado, uma melhoria após o ajuste, uma falha do especialista e um caso fora do escopo. Use o teste apenas após congelar as escolhas. Modelos podem ter visto benchmarks públicos no pré-treinamento; a separação feita aqui impede vazamento no **seu fine-tuning**, mas não certifica ausência de contaminação anterior.

O idioma dos datasets é inglês. Apresente os resultados quantitativos em inglês. Uma demonstração em português pode ser incluída separadamente como teste de transferência, sem afirmar que houve especialização em português. Traduzir todo o dataset mudaria o experimento e exigiria nova revisão.

## 10. Problemas frequentes

| Sintoma | Ação concreta |
|---|---|
| `CUDA out of memory` | Pare o treino, confira processos em `nvidia-smi`, mantenha microbatch 1 e refaça a preparação com `--max-seq 1536` ou `1024`. Isso reduz cobertura: use os novos dados em todas as condições. Acumulação de gradientes não reduz a memória de uma sequência individual. |
| Ainda não cabe | Reduza `--rank` para 8; considere limitar LoRA a `q_proj/k_proj/v_proj/o_proj` como nova configuração documentada. Evite trocar silenciosamente o modelo durante a comparação. |
| Travamento/erro NCCL somente com duas GPUs | Confira `nvidia-smi topo -m` e teste uma GPU. Para diagnóstico de P2P, tente apenas nessa execução `NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun ...`. Pode reduzir desempenho. Não altere IOMMU/BIOS como primeira medida. |
| Apenas uma GPU usada | Use o comando `torchrun --nproc-per-node=2 --`; a inferência com `predict` é deliberadamente de uma GPU. |
| Perda `nan`/`inf` | Confira precisão escolhida, reduza learning rate e investigue o log; não continue treinando esperando correção espontânea. |
| Erro de acesso ao xLAM | Entre no Hugging Face, aceite as condições do dataset e execute `hf auth login` no mesmo usuário/ambiente. |
| Muitos casos fora do contexto | Inspecione `length_report.json`. Aumentar contexto eleva memória; um esquema/catálogo mais compacto deve usar regras disponíveis na inferência, nunca tabelas/funções reveladas pela resposta correta. |
| Bom treino, validação pior | Reduza épocas/LR e confira o mascaramento da perda, separação por grupos e diversidade. Não altere o teste para melhorar o resultado. |
| JSON/SQL truncado | Ajuste `--max-new` e a reserva `--max-new` na preparação de forma coordenada; refaça todas as condições. Use outro diretório de dados e resultados. |
| Modelos geram explicações | Verifique que está usando o mesmo template, `enable_thinking=False`, prompt e revisão registrados. Não acrescente um pós-processamento inteligente só ao especialista. |

O script usa um mapa explícito `device_map={"": LOCAL_RANK}` para treinar. Não o substitua por `device_map="auto"` em DDP: esse modo pode espalhar um modelo por GPUs e conflitar com um processo por placa.

## 11. Roteiro para a apresentação de uma hora

| Minutos | Demonstração |
|---|---|
| 0–7 | Problema, aplicação real, hipótese e baseline |
| 7–15 | Dataset pronto, estrutura de um exemplo e divisão dos conjuntos |
| 15–25 | Quantização, LoRA, perda somente na resposta e distribuição entre GPUs |
| 25–35 | Comandos utilizados, logs, memória e curva de validação do treino já concluído |
| 35–48 | Comparação completa e demonstração de três ou quatro casos |
| 48–55 | Falhas, custo, limites e como integrar |
| 55–60 | Perguntas |

O treinamento completo é feito antes. Leve adaptador, cache do modelo, datasets e resultados em disco. A parte local deve funcionar sem internet quando os arquivos já estiverem no cache; o LLM por API deve ter suas respostas previamente registradas para a aula não depender da conexão.

Se mostrar **os dois laboratórios na mesma hora**, use aproximadamente 10 minutos de fundamentos comuns, 20 minutos de ferramentas, 20 minutos de SQL e 10 minutos de comparação e perguntas.

## 12. O que preservar ao terminar

Guarde o Markdown, scripts utilizados, `requirements.txt`, `requirements-resolved.txt`, `revisions.json`, relatórios de preparação, JSONLs utilizados, logs, `run_report.json`, adaptador final, predições, metadados de inferência e métricas. Registre a versão do driver e qual GPU serviu cada medição.

O adaptador sozinho não é um modelo completo. Para reutilizá-lo, carregue a mesma revisão do Qwen base e depois o adaptador, como faz `load_model`. Não é necessário mesclar os pesos para a demonstração. Uma futura conversão para GGUF/Ollama deve ser tratada como outro artefato, com nova avaliação, porque a quantização e o runtime mudam.

**Validação deste tutorial:** sintaxe dos scripts e lógica de avaliação/preparação verificadas com exemplos controlados. O treinamento CUDA, os downloads completos e as chamadas a um LLM pago não foram executados no ambiente de elaboração. Os testes curtos acima são as verificações necessárias no seu laboratório; não há acurácia, duração ou consumo de VRAM inventados neste documento.

## 13. Referências primárias

Fontes consultadas em 10/10/2026; preserve as revisões registradas em seu laboratório.

- [Salesforce, dataset xLAM/APIGen](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k).
- [APIGen, artigo original](https://arxiv.org/abs/2406.18518).
- [Salesforce, repositório xLAM](https://github.com/SalesforceAIResearch/xLAM).
- [Berkeley, BFCL e diferença entre avaliações single-turn e multi-turn](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html).

- [Qwen3-1.7B, modelo, template e modo sem thinking](https://huggingface.co/Qwen/Qwen3-1.7B).
- [PEFT 0.17, quantização e QLoRA](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization).
- [Transformers 4.55.4, Trainer e TrainingArguments](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer).
- [PyTorch, instalações de versões anteriores](https://pytorch.org/get-started/previous-versions/).
- [bitsandbytes 0.47, requisitos de instalação](https://huggingface.co/docs/bitsandbytes/v0.47.0/en/installation).
- [Hugging Face Hub 0.34.4, autenticação e CLI](https://huggingface.co/docs/huggingface_hub/v0.34.4/en/guides/cli).
- [OpenAI, Chat Completions](https://platform.openai.com/docs/api-reference/chat/create).
- [OpenAI, gpt-4.1-mini e preços](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

As referências OpenAI foram verificadas em 13/10/2026.

Scripts auxiliares: [inspecionar_dados.py](scripts/lab01-ferramentas/inspecionar_dados.py), [listar_checkpoints.py](scripts/lab01-ferramentas/listar_checkpoints.py), [preparar_demo.py](scripts/lab01-ferramentas/preparar_demo.py), [count_api_calls.py](scripts/lab01-ferramentas/count_api_calls.py).

## Executar os comandos por script Bash

O arquivo [laboratorio-01-slm-ferramentas-xlam.sh](laboratorio-01-slm-ferramentas-xlam.sh) reúne os 18 blocos Bash deste roteiro em etapas numeradas. Liste as etapas e execute um intervalo:

```bash
bash ~/git/slm-lab/laboratorio-01-slm-ferramentas-xlam.sh --list
bash ~/git/slm-lab/laboratorio-01-slm-ferramentas-xlam.sh --steps 1-5
```

Ajuste o caminho do repositório se necessário. Execute as etapas seguintes em ordem conforme o roteiro; selecione as alternativas adequadas. Configuração de API e chamadas devem ficar na mesma execução. O servidor do laboratório 04 ocupa o terminal, e a chamada HTTP deve ser feita em outro terminal. Downloads manuais e caminhos de exemplo continuam exigindo os ajustes descritos no texto.

Cada execução grava stdout e stderr em `logs/laboratorio-01-slm-ferramentas-xlam-DATA-HORA-PID.log` na pasta do laboratório. Os logs específicos dos comandos também são mantidos. Uma falha interrompe o script e registra o código de saída. O script usa a pasta do laboratório indicada neste roteiro; `LAB_WORK_DIR` permite escolher outra.
