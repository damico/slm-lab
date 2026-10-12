# Laboratório 04, SLM especialista em decisões com ABCD

**Objetivo:** treinar e servir localmente um modelo que recebe o histórico de atendimento e escolhe o próximo passo entre alternativas conhecidas, retornando probabilidades e uma indicação de revisão.

**Hardware-alvo:** duas RTX 3070 de **8 GB cada**. **Base:** Qwen3-1.7B. **Método:** classificação de sequência com QLoRA e calibração posterior. **Dataset público:** ABCD. **Aplicação:** CLI e API HTTP local em `/decide`.

Este é o quarto laboratório da série. É adequado à apresentação técnica de uma hora: você executa preparação, treino e avaliação com antecedência, e demonstra a decisão e a API durante a sessão.

## 1. O resultado utilizável

A aplicação recebe eventos anteriores de um atendimento e responde com:

- `choice`: próxima ação recomendada, ou `retrieve_utterance` quando é preciso responder/perguntar ao cliente.
- `probabilities`: distribuição estimada sobre o catálogo fixo de alternativas.
- `confidence`: maior probabilidade após o ajuste de temperatura.
- `requires_review`: indica se a probabilidade ficou abaixo do limiar escolhido em dados separados.
- `policy_ids` e `dropped_turns`: permitem inspecionar os trechos de regras recuperados e o uso de janela de histórico.

O código consumidor pode exibir a recomendação para um atendente ou escolher a próxima etapa de um fluxo. A API deste lab devolve a decisão; a execução, os argumentos das ações e as regras de autorização pertencem à aplicação integradora.

Um pedido de devolução pode exigir primeiro consultar a conta, validar a compra ou coletar informações. O desafio é escolher a etapa apropriada usando o que já foi observado. As probabilidades ajudam a decidir quando encaminhar o caso para revisão, desde que sua qualidade seja medida.

### O sentido de “Jev-like” neste laboratório

A inspiração é a interface **estado → escolha entre opções → probabilidades**, documentada pela TypeSafe para o Jev. Aqui usamos um modelo aberto com uma camada de classificação. Não reproduzimos a arquitetura, o treinamento RLCD nem os resultados do Jev.

O catálogo é fixo. Adicionar uma ação nova exige adaptar e treinar a camada de saída. Oferecer alternativas arbitrárias a cada chamada exigiria outro desenho, como um modelo que pontua pares cenário–opção. Este roteiro não implementa essa segunda modalidade.

## 2. O que foi conferido no ABCD

O repositório oficial é [asappresearch/abcd](https://github.com/asappresearch/abcd), com licença MIT e diálogos humanos em cenários simulados de atendimento. A revisão usada é:

```text
6b8700ce67c6b37b062dd7a60abc76d7ef832a97
```

A inspeção do arquivo `abcd_v1.1.json.gz` dessa revisão encontrou:

| Split original | Conversas | Turnos com ação | Turnos em que o agente fala |
|---|---:|---:|---:|
| Treino | 8.034 | 29.190 | 75.985 |
| Desenvolvimento | 1.004 | 3.684 | 9.600 |
| Teste | 1.004 | 3.608 | 9.544 |

O catálogo tem 30 ações. **A documentação também prevê `end_conversation`, mas esse rótulo não aparece nos exemplos deste arquivo.** Por isso, o classificador tem **31 classes: as 30 ações e `retrieve_utterance`**. Encerramento fica fora do modelo. Essa é uma correção do desenho preliminar, baseada nos dados reais.

O corpus inclui intenções e cenários usados na coleta. Eles não entram no prompt: revelariam informação que a aplicação ainda precisa inferir. Usamos textos originais, anotações de ações anteriores já executadas e trechos das regras públicas. Não há anotação manual de novos exemplos.

São atendimentos em inglês e políticas de uma empresa fictícia. O experimento valida esse domínio; português, novas empresas e novas políticas exigem avaliação própria. O gabarito é o passo registrado na conversa, não uma prova de que somente esse passo seria aceitável. O resultado do lab não é o score oficial completo de AST/CDS, pois não avalia argumentos, seleção da frase nem sucesso final de uma conversa autônoma.

## 3. Desenho técnico e prevenção de vazamento

O Qwen processa o contexto e a camada `score` produz 31 logits. A perda supervisionada é entropia cruzada sobre o rótulo correto. Os pesos quantizados da base ficam congelados; treinamos LoRA e a camada de classificação, que permanece em ponto flutuante e é salva junto do adaptador.

**O texto do turno-alvo nunca entra na entrada daquele exemplo.** As anotações `targets` descrevem o próprio turno, e não o turno seguinte. O conversor emite o exemplo antes de adicionar o evento ao histórico. Isso evita, por exemplo, prever `offer-refund` lendo uma mensagem que já anuncia o reembolso.

As conversas são mantidas inteiras no mesmo split. O treino original fornece os exemplos de ajuste; o desenvolvimento é dividido por conversa para finalidades distintas:

| Conjunto do lab | Origem | Uso |
|---|---|---|
| `train` | Até 2.000 conversas do treino, por padrão | Ajustar LoRA e camada de saída |
| `validation` | 502 conversas do desenvolvimento | Selecionar checkpoint/hiperparâmetros |
| `calibration` | 251 conversas do desenvolvimento | Ajustar uma temperatura |
| `selection` | 251 conversas do desenvolvimento | Escolher limiar para revisão |
| `test` | 1.004 conversas do teste original | Medir tudo depois de congelar a receita |

O limite de 2.000 conversas é um ponto de partida para o laboratório. Para usar todo o treino, há um comando específico adiante. Duplicatas de conversas completas de dev/test são removidas do treino; duplicatas internas do treino também. Saudações e outros prefixos comuns podem continuar aparecendo em conversas diferentes.

### Como as regras entram na decisão

O script transforma as regras públicas em 255 trechos e usa TF-IDF para recuperar os mais relacionados às últimas mensagens. A consulta não recebe a intenção correta nem o subfluxo do gabarito. O recuperador é fixo e compartilhado por todas as condições.

O orçamento inicial é de 400 tokens para regras dentro de 2.048 tokens totais. Só entram trechos completos. Se o contexto ultrapassar o limite, os eventos mais antigos são retirados da janela, mantendo uma lista dos nomes de ações já observadas. Isso preserva a existência de uma ação anterior, mas pode perder seus argumentos, a ordem de repetições e outros detalhes; `dropped_turns` permite medir essa limitação. Se nem o evento mais recente couber, o caso é marcado como inelegível e encaminhado para revisão.

Esse conjunto combina adaptação do modelo e recuperação de regras. O comparativo mede a solução completa com entradas iguais. Para atribuir causalmente um ganho apenas ao LoRA, seria necessário um controle adicional com a mesma camada de classificação treinada sobre a base congelada, sem LoRA.

## 4. Preparar o ambiente nas duas placas de 8 GB

Use Ubuntu, Bash, Python 3.11 ou 3.12, aproximadamente 32 GB de RAM e uma reserva inicial de 30–45 GB em disco. O dataset descompactado e os JSONLs com históricos ocupam mais espaço que o download. Esses valores são planejamento, não medições de treino nas suas GPUs.

**O DDP não soma os 8 GB das placas.** Cada GPU recebe uma cópia quantizada e processa seus exemplos. O modelo e as ativações precisam caber em uma placa. O ganho de velocidade da segunda GPU depende da comunicação e deve ser medido.

Prepare primeiro o ambiente compartilhado conforme [Preparação geral](preparacao-geral.md).

Crie uma pasta independente dos outros labs e copie este Markdown para ela com o nome `laboratorio.md`. O exemplo considera o repositório em `~/git/slm-lab`; ajuste o caminho de origem se você o salvou em outro lugar:

A cópia do Markdown para `./laboratorio.md` deixa uma referência do roteiro na pasta do experimento. Ela é opcional para executar os scripts, que agora estão em arquivos `.py` separados.

```bash
mkdir -p lab04-decisoes
cd lab04-decisoes
cp ~/git/slm-lab/laboratorio-04-slm-decisoes-abcd.md ./laboratorio.md
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
python3 --version
```

As cópias de `~/slm-labs/requirements.txt` e `~/slm-labs/requirements-resolved.txt` para a pasta atual registram, junto dos resultados, a receita de dependências e as versões efetivamente instaladas. Elas não criam outro venv nem instalam pacotes.

```bash
source ~/slm-labs/.venv/bin/activate
cp ~/slm-labs/requirements.txt ./requirements.txt
cp ~/slm-labs/requirements-resolved.txt ./requirements-resolved.txt
mkdir -p raw runs results logs
export TOKENIZERS_PARALLELISM=false
```

As versões são fixadas para reprodutibilidade, não por serem as mais recentes. O wheel cu128 exige driver compatível; consulte a distribuição oficial do PyTorch se CUDA falhar. Os comandos são para Bash, não para PowerShell. Em Windows, use um ambiente Linux/WSL2 com as GPUs visíveis.

```bash
python ~/git/slm-lab/verificar_ambiente.py
```

## 5. Copiar os scripts do laboratório

Os scripts estão em [scripts/lab04-decisoes](scripts/lab04-decisoes). Copie-os para a pasta de trabalho com o comando abaixo. Ajuste o caminho do repositório se necessário. `cp -n` preserva arquivos existentes; compare cópias antigas antes de atualizá-las. O Markdown é apenas o roteiro e não precisa ser lido por um extrator.

`cp -n` copia os scripts da pasta específica deste laboratório no repositório para a pasta atual. Isso permite executar os comandos `python arquivo.py` e resolver os imports locais, mantendo dados e resultados isolados dos outros laboratórios. A opção `-n` preserva arquivos de destino que já existem.

```bash
cp -n ~/git/slm-lab/scripts/lab04-decisoes/*.py ./
```

## 6. Baixar e preparar os dados

O comando `prepare_abcd.py` baixa quatro arquivos da revisão fixada do repositório ABCD no GitHub: `abcd_v1.1.json.gz` contém as conversas e ações que viram exemplos e gabaritos; `ontology.json` define o catálogo de ações; `guidelines.json` fornece as regras recuperadas para compor o contexto; `LICENSE` preserva os termos de uso do material. Os arquivos ficam em `raw/abcd` por padrão. O tokenizer do Qwen3-1.7B vem do Hugging Face e permite montar a janela de histórico dentro do limite de tokens.

```bash
set -o pipefail
python prepare_abcd.py --out data --train-conversations 2000 \
  --max-seq 2048 --policy-tokens 400 2>&1 | tee logs/prepare.log
```

O script baixa os dados da revisão fixada, registra a revisão do modelo em `model_revision.json`, monta as entradas e cria os cinco splits. A preparação usa CPU e o tokenizer; os pesos do modelo serão baixados no treino/inferência. Use uma pasta `--raw` exclusiva deste lab; os arquivos em cache não são baixados novamente.

Verifique os tamanhos, a distribuição de classes e os limites:

```bash
python inspecionar_dados.py
```

`data/bundle.json` contém a ordem das classes, as regras e a configuração de construção do contexto. Preserve-o junto do adaptador e da calibração. Alterar o catálogo ou seu ordenamento invalida a camada de classificação.

Para o treino completo, prepare outra pasta e use-a em **todos** os comandos subsequentes com `--data data_full`, além de ajustar os caminhos dos JSONLs:

A preparação completa abaixo usa as mesmas fontes externas e o mesmo tokenizer da preparação anterior, reaproveitando arquivos disponíveis no cache. Sua finalidade é gerar exemplos a partir de todas as conversas de treino, em uma pasta de saída diferente.

```bash
python prepare_abcd.py --out data_full --train-conversations 0 \
  --max-seq 2048 --policy-tokens 400
```

Escolha o tamanho do treino pela validação, antes de abrir o teste. Se uma classe não estiver presente no subconjunto de treino, o script pede para aumentar o número de conversas. Os registros inelegíveis permanecem no teste; são contados como falhas na acurácia global e como revisão no uso local.

## 7. Teste curto e treinamento

O runtime inicializa o grupo NCCL vinculando cada processo à GPU indicada por `LOCAL_RANK`, com timeout de cinco minutos para operações coletivas. Isso torna explícita a escolha de GPU usada nas barreiras, conforme a [documentação do PyTorch 2.7](https://docs.pytorch.org/docs/2.7/distributed.html). A barra em 100% indica o fim dos passos, mas o treino só está concluído quando retorna ao terminal e grava `final_adapter` e `run_report.json`. Mensagens adicionais identificam a saída do Trainer e a gravação final. Se atualizar o runtime no repositório, substitua também a cópia local de `decision_lab.py`; `cp -n` preserva cópias antigas.

Nos comandos `torchrun`, o separador `--` antes do nome do script encerra as opções do lançador. Assim, `--run` e os demais argumentos seguintes chegam ao script de treino sem serem interpretados como abreviações de opções do `torchrun`.

Na primeira execução do treino ou da inferência local, o runtime baixa do Hugging Face os pesos e a configuração do Qwen3-1.7B que ainda não estiverem no cache. Esses arquivos são a base pré-treinada sobre a qual o laboratório ajusta os adaptadores e, no laboratório 04, a camada de classificação. Inferências posteriores reutilizam o cache e carregam o adaptador da pasta local indicada; se algum arquivo do tokenizer ou da base estiver ausente, ele também é obtido do Hugging Face.

Configuração inicial: rank 16, alpha 32, dropout 0,05, learning rate `1e-4`, duas épocas, microbatch 1 e acumulação 8. Com duas GPUs, o batch efetivo é 16 decisões por atualização. O runtime usa NF4, dupla quantização, SDPA, gradient checkpointing e BF16 quando suportado, ou FP16.

Uma mensagem dizendo que `score.weight` foi inicializado é esperada na criação da base classificadora: essa camada não existe no checkpoint de linguagem e precisa ser aprendida. O adaptador salvo inclui a camada treinada por meio de `modules_to_save=["score"]`. Recarregar somente os pesos da base produziria uma classificação sem treinamento.

```bash
CUDA_VISIBLE_DEVICES=1 python decision_lab.py train --data data \
  --run runs/smoke_1gpu --max-steps 10 --accum 1 --eval-steps 5 --eval-limit 16 \
  2>&1 | tee logs/smoke_1gpu.log
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -- \
  decision_lab.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-steps 10 --eval-limit 32 \
  2>&1 | tee logs/smoke_2gpu.log
```

O treino principal começa novamente da base original, em outra pasta:

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_train.csv &
LAB04_MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  torchrun --standalone --nproc-per-node=2 -- decision_lab.py train \
  --data data --run runs/main --epochs 2 --accum 8 --rank 16 --lr 0.0001 \
  --eval-steps 100 --eval-limit 512 2>&1 | tee logs/main_train.log
kill "$LAB04_MONITOR_PID"
```

A cada 100 passos, o Trainer avalia a perda em até 512 decisões fixas da validação. Ao terminar, carrega o melhor checkpoint avaliado e salva `runs/main/final_adapter`. O relatório de treino registra tempo e pico de memória alocada no processo principal; o CSV do `nvidia-smi` registra ambas as placas.

**Medição nesta máquina:** o treino principal terminou com 3.270 passos, duas épocas e `train_runtime` de 30.062 s (8h21min). `/usr/bin/time` mediu 8h26min21s de tempo total e reportou código de saída 0. O adaptador e `run_report.json` foram salvos. O pico de VRAM não foi fornecido nos resultados disponíveis; não estimamos esse valor a partir da memória RAM do processo.

Se faltar memória, feche outras inferências, reduza rank para 8 e, se necessário, prepare novos dados com `--max-seq 1536`. Mantenha o mesmo contexto em todas as condições comparadas. Aumentar a acumulação não resolve um único exemplo que excede a memória. Uma execução em uma GPU pode usar `--accum 16` para preservar o batch efetivo. Para retomar, use `--resume runs/main/checkpoint-N` com um checkpoint existente e a mesma receita.

## 8. Validar o especialista e treinar uma referência simples

```bash
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/validation.jsonl --output results/val_specialist.jsonl
python evaluate_decisions.py score data/validation.jsonl results/val_specialist.jsonl \
  --out results/val_specialist
```

Inspecione `*.cases.jsonl` e os erros por classe. Como responder ao cliente é frequente, a acurácia global sozinha pode enganar. Dê destaque à macro-F1 e à acurácia somente nos casos cujo gabarito é uma das 30 ações.

Inclua uma referência barata, TF-IDF + regressão logística, treinada nos mesmos exemplos e com os mesmos textos. Isso ajuda a verificar se a complexidade do SLM se justifica:

```bash
python decision_lab.py baseline --data data --output runs/tfidf.joblib
python decision_lab.py predict --data data --backend tfidf --artifact runs/tfidf.joblib \
  --input data/validation.jsonl --output results/val_tfidf.jsonl
python evaluate_decisions.py score data/validation.jsonl results/val_tfidf.jsonl \
  --out results/val_tfidf
```

Antes da calibração, congele o adaptador, o contexto e as regras. Se trocar o modelo depois, repita a calibração e a seleção de limiar com as mesmas separações de dados.

## 9. Calibrar probabilidades e escolher quando revisar

Usamos `softmax(logits / T)`. A temperatura positiva mantém a opção de maior pontuação; ela pode melhorar as probabilidades, **mas não muda a acurácia top-1**. A calibração é um resultado empírico a medir, não uma garantia de que o modelo reconhecerá qualquer situação desconhecida.

Primeiro, gere logits para o conjunto reservado de calibração e ajuste T minimizando a perda logarítmica:

```bash
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/calibration.jsonl --output results/calibration_logits.jsonl
python evaluate_decisions.py calibrate data/calibration.jsonl results/calibration_logits.jsonl \
  --out runs/temperature.json
```

Depois, escolha o limiar em **outras conversas**, do conjunto `selection`:

```bash
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/selection.jsonl --output results/selection_logits.jsonl
python evaluate_decisions.py select data/selection.jsonl results/selection_logits.jsonl \
  --calibration runs/temperature.json --target-error 0.10 --min-selected 100 \
  --out runs/decision_policy.json
```

O critério procura a maior cobertura com pelo menos 100 decisões e erro observado de até 10% nesse conjunto. **10% é uma escolha didática, não um nível recomendado para autorizar reembolsos.** O limiar é global e seu resultado pode variar por classe. Se nenhum ponto atender ao critério, o arquivo usa limiar 1,01 para encaminhar todos os casos à revisão.

A escolha é empírica e não garante erro máximo no futuro. No teste, reporte novamente cobertura e erro dos casos selecionados. Confira também classes raras. Para uso real, o custo de errar cada ação deve orientar critérios mais específicos.

`requires_review` é uma decisão do código a partir das probabilidades. Não é uma classe de “caso desconhecido” ensinada pelo ABCD. O modelo sempre distribui probabilidade entre o catálogo conhecido e pode ficar confiante fora do domínio.

Os arquivos de calibração registram hashes e identidade do adaptador. Os scripts recusam aplicá-los a outro modelo ou configuração de entrada.

## 10. Comparar com o modelo original e um LLM maior

Execute o teste depois de congelar tudo. Todos recebem os mesmos históricos e trechos de regras. O especialista retorna uma classe por passagem; as referências generativas recebem a instrução de responder somente com o rótulo, sem justificativa.

```bash
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data \
  --backend classifier --artifact runs/main/final_adapter \
  --input data/test.jsonl --output results/test_specialist.jsonl
CUDA_VISIBLE_DEVICES=0 python decision_lab.py predict --data data --backend base \
  --input data/test.jsonl --output results/test_original.jsonl
python decision_lab.py predict --data data --backend tfidf --artifact runs/tfidf.joblib \
  --input data/test.jsonl --output results/test_tfidf.jsonl
```

O original é Qwen3-1.7B com sua cabeça de linguagem, usando prompt. A diferença para o especialista inclui tanto o treinamento quanto a cabeça de saída. Não compare com uma cabeça de classificação aleatória e a apresente como “modelo original”.

A predição original é determinística (`do_sample=False`). O script normaliza `temperature`, `top_p` e `top_k` herdados da configuração do Qwen para evitar avisos de parâmetros de amostragem que não se aplicam à geração gulosa. Se você já copiou `decision_lab.py` para a pasta do laboratório, atualize essa cópia a partir de `scripts/lab04-decisoes/decision_lab.py` antes de repetir a predição.

Para comparar com um modelo servido pela OpenAI, configure o endpoint oficial de Chat Completions, o identificador do modelo e uma chave da API Platform. Este exemplo usa `gpt-4.1-mini`; a API é cobrada conforme o uso e o modelo é executado na nuvem, não nas RTX 3070. Consulte a [documentação de Chat Completions](https://platform.openai.com/docs/api-reference/chat) e a [lista de modelos](https://platform.openai.com/docs/models) para modelos e parâmetros disponíveis na sua conta. Antes de chamar o endpoint, conte quantas linhas elegíveis serão enviadas:

```bash
python count_api_calls.py
```

O probe abaixo limita a validação às três primeiras linhas. Como linhas inelegíveis não geram requisição, o número pode ficar entre zero e três. O teste completo faz uma chamada para cada linha elegível de `data/test.jsonl`. O total real pode crescer se houver retries automáticos por erros HTTP temporários; o prompt inteiro de cada exemplo é enviado à OpenAI.

**Referência de custo e duração observada nesta configuração:** no teste ABCD, `data/test.jsonl` tinha 13.152 linhas elegíveis. Em uma execução real com `gpt-4.1-mini`, 4.274 chamadas levaram 58min44s (cerca de 73 chamadas/minuto), com 3.244.944 tokens de entrada e 14.322 de saída reportados. Extrapolando a média para as 13.152 chamadas, a etapa completa levaria cerca de **3h** e custaria aproximadamente **US$ 4,06** pelas tarifas consultadas em 12/10/2026 (US$ 0,40/1M tokens de entrada e US$ 1,60/1M tokens de saída). O tempo varia com a latência da rede e o serviço; o custo varia com o tamanho dos prompts e retries. Para uma execução nova, use o contador e o uso reportado em `results/test_llm.jsonl`; a estimativa histórica não é uma cotação.

Configure a URL, modelo e chave:

```bash
export LLM_URL='https://api.openai.com/v1/chat/completions'
export LLM_MODEL='gpt-4.1-mini'
read -r -s -p 'Chave da API OpenAI: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"max_tokens":32}'
python decision_lab.py predict --data data --backend api \
  --input data/validation.jsonl --output results/val_api_probe.jsonl --limit 3
```

O `max_tokens` limita o tamanho da resposta, que deve conter apenas um rótulo. A chave é lida sem eco no terminal; não a coloque nos comandos salvos. Faça primeiro o probe curto e confira as saídas antes do conjunto de teste.

Depois de verificar que as respostas da API contêm apenas rótulos completos:

```bash
python decision_lab.py predict --data data --backend api \
  --input data/test.jsonl --output results/test_llm.jsonl
python evaluate_decisions.py score data/test.jsonl results/test_specialist.jsonl \
  --calibration runs/decision_policy.json --out results/specialist
python evaluate_decisions.py score data/test.jsonl results/test_original.jsonl --out results/original
python evaluate_decisions.py score data/test.jsonl results/test_tfidf.jsonl --out results/tfidf
python evaluate_decisions.py score data/test.jsonl results/test_llm.jsonl --out results/llm
python evaluate_decisions.py compare results/original.cases.jsonl results/specialist.cases.jsonl
python evaluate_decisions.py compare results/llm.cases.jsonl results/specialist.cases.jsonl
```

Os arquivos de predição existentes não são sobrescritos. Para repetir, use nomes novos. Predições ausentes ou inválidas contam como erros; não apresente uma execução com `--limit` como teste completo. As probabilidades das referências generativas não são estimadas por texto: NLL, Brier e ECE ficam indisponíveis para elas neste cliente.

### Como ler os relatórios

| Métrica | Significado |
|---|---|
| `accuracy_all` | Acerto em todas as decisões, incluindo casos que não couberam |
| `macro_f1` | Média de F1 das 31 classes, dando peso igual a cada classe |
| `action_only_accuracy` | Acerto quando o gabarito exige uma das 30 ações, incluindo como erro responder quando deveria agir |
| `per_class` / `confusion.json` | Quais ações são confundidas e seu suporte |
| `probabilities_before/after` | NLL, Brier somado nas classes, ECE em 10 intervalos e dados de confiabilidade |
| `coverage_all` / `selected_error` | Parcela selecionada pelo limiar e erro observado nessa parcela |
| `p50_s` / `p95_s` | Tempo por previsão com status `ok`, após aquecimento local |

As latências incluem tokenização e inferência; excluem construção/recuperação do contexto, carga do modelo e avaliação. A API inclui rede. Mostre também a quantidade de respostas bem-sucedidas, pois os percentis excluem falhas. O servidor local executa a recuperação de regras por pedido, portanto seu tempo total não será exatamente o do benchmark.

O bootstrap compara B − A reamostrando **conversas inteiras**, porque decisões de uma mesma conversa são correlacionadas. O intervalo de 95% é aproximado. Se incluir zero, não apresente o ganho de acurácia como estabelecido pelo teste.

Resultados do teste completo, com 13.152 decisões em 1.004 conversas:

| Condição | Acurácia | Macro-F1 | Acerto nas ações | p50/p95 por decisão | Memória/custo |
|---|---|---|---|---|---|
| TF-IDF + regressão logística | 75,87% | 0,1006 | 16,52% | 0,0028 / 0,0032 s | CPU; sem cobrança por chamada |
| Qwen3-1.7B original com prompt | 65,13% | 0,0316 | 0,94% | 0,3454 / 0,3802 s | GPU; pico não registrado |
| Qwen3-1.7B especializado | **85,45%** | **0,6812** | **64,33%** | 0,1629 / 0,2101 s | GPU; pico não registrado |
| OpenAI `gpt-4.1-mini` | 55,37% | 0,2445 | 40,44% | 0,8188 / 1,0552 s | API; aproximadamente US$ 4,04 |

O bootstrap pareado por conversa estimou ganho de 20,32 pontos percentuais do especialista sobre o modelo original (IC 95%: 19,37 a 21,28) e 30,08 pontos sobre a API (IC 95%: 29,15 a 31,04). O especialista selecionou 11.490 casos para não revisão, cobertura de 87,36%, com erro observado de 10,08% no teste. O alvo de 10% foi escolhido didaticamente na seleção; esse resultado de teste não garante o mesmo erro em produção.

O teste teve 13.135 respostas com latência bem-sucedida para a API e 17 sem sucesso; no modelo original, 13.120 tiveram latência bem-sucedida. As acurácias usam todas as 13.152 decisões, penalizando falhas. Investigue os status em `results/test_llm.jsonl` e `results/test_original.jsonl` antes de atribuir todas essas falhas à qualidade do modelo.

A API reportou 9.931.657 tokens de entrada e 44.131 de saída. Com as tarifas consultadas em 12/10/2026 para `gpt-4.1-mini` (US$ 0,40 por milhão de entrada e US$ 1,60 por milhão de saída), o custo calculado é aproximadamente US$ 4,04, sem incluir eventuais tributos. Consulte a [página de preços atual](https://developers.openai.com/api/docs/models/gpt-4.1-mini) ao repetir o experimento.

## 11. Usar a CLI e a API local

Para experimentar pelo navegador com timeline, probabilidades e regras recuperadas, use o [avaliador web](avaliador/README.md). Ele utiliza o mesmo adaptador e a mesma calibração desta seção, com comparação OpenAI opcional. O README inclui os dois comandos de inicialização e a configuração das portas.

A preparação separa doze entradas de teste para demonstração, pela ordem do arquivo e antes de conhecer os acertos. Os JSONs de entrada não contêm o gabarito. Os rótulos esperados ficam em `data/demo_expected.json` para conferência.

```bash
CUDA_VISIBLE_DEVICES=0 python serve_decisions.py --data data \
  --adapter runs/main/final_adapter --calibration runs/decision_policy.json \
  --input data/demo/case_00.json
```

Para manter o modelo carregado e atender pedidos, rode no primeiro terminal:

```bash
CUDA_VISIBLE_DEVICES=0 python serve_decisions.py --data data \
  --adapter runs/main/final_adapter --calibration runs/decision_policy.json --port 8084
```

Em outro terminal, dentro da pasta do lab:

```bash
curl --fail-with-body -sS http://127.0.0.1:8084/decide \
  -H 'Content-Type: application/json' \
  --data-binary @data/demo/case_00.json
```

A API fica somente em `127.0.0.1` e processa pedidos em sequência, suficiente para o laboratório. Essa não é uma implantação de produção. Um serviço Java, Python ou um componente do InfiniteStack no mesmo host pode consumir esse contrato.

Para montar seu próprio caso, envie um objeto `history` com eventos já observados. Exemplo sintético de integração, sem acerto garantido:

```bash
cat > meu_caso.json <<'JSON'
{
  "history": [
    {"role":"agent","text":"How can I help you?"},
    {"role":"customer","text":"I want to return a shirt because it is the wrong size."},
    {"role":"agent","text":"May I have your full name?"},
    {"role":"customer","text":"Alex Example"}
  ]
}
JSON
curl --fail-with-body -sS http://127.0.0.1:8084/decide \
  -H 'Content-Type: application/json' --data-binary @meu_caso.json
```

Se uma ação foi realmente executada, seu evento anterior pode incluir `action`, usando um nome do catálogo, e `text` com o resultado observado. Não adicione uma recomendação ao histórico como se já tivesse sido executada. Não envie `scenario`, `targets`, `label` ou o resultado futuro; o contrato aceita somente os eventos anteriores.

`retrieve_utterance` significa que o atendente ou outro componente deve conversar/coletar informação. O modelo não produz essa frase, não preenche argumentos, não encerra o atendimento e não prova que todas as precondições de uma ação foram atendidas. O resultado é uma recomendação operacional entre as opções treinadas.

## 12. Roteiro de apresentação e limites do resultado

| Minutos | Demonstração |
|---|---|
| 0–8 | Estado do atendimento, opções e uma chamada real à API |
| 8–18 | Dataset, decisão antes do turno-alvo e separação por conversa |
| 18–30 | Cabeça de classificação, LoRA, 4 bits e duas GPUs de 8 GB |
| 30–42 | Comparação com original, TF-IDF e LLM maior |
| 42–52 | Calibração: confiança, erro e proporção de revisões |
| 52–60 | Caso novo, erros representativos e perguntas |

Se compartilhar a hora com SQL, reduza este caso a uma demonstração de 10–15 minutos e uma tabela de resultados. Treino e benchmark ficam prontos antes. Leve os adaptadores, bundle, calibração, entradas e respostas gravadas para não depender do endpoint remoto na apresentação.

A avaliação usa o histórico anterior registrado pelos humanos, incluindo ações anteriores corretas. Isso é avaliação da próxima decisão com histórico de referência; não mede uma conversa inteira em que erros anteriores do próprio modelo se acumulam. Sucesso do fluxo completo é outro experimento.

Este dataset público pode ter aparecido no pré-treinamento dos modelos. A divisão por conversa reduz vazamento do seu ajuste, mas não elimina essa possibilidade. Alterações de política, linguagem e catálogo também podem degradar a calibração e exigem novos testes.

## 13. Verificações realizadas e referências

Na revisão indicada, foram inspecionadas as conversas do ABCD, confirmando 131.611 decisões e as 31 classes. O treino, a inferência e a avaliação descritos acima foram executados nas duas RTX 3070 e na API OpenAI. A qualidade e os tempos da tabela são medições dessa execução e desse split; o pico de VRAM não foi registrado nos dados disponíveis. Os resultados não validam outros domínios nem garantem desempenho futuro.

Fontes primárias consultadas em 12/10/2026:

- [ASAPP, repositório oficial ABCD, dados e licença](https://github.com/asappresearch/abcd).
- [Chen et al., artigo ABCD, NAACL 2021](https://aclanthology.org/2021.naacl-main.239/).
- [TypeSafe, interface Choice do Jev](https://docs.typesafe.ai/primitives/choice).
- [Qwen3-1.7B, modelo aberto utilizado](https://huggingface.co/Qwen/Qwen3-1.7B).
- [Transformers 4.55.4, Qwen3ForSequenceClassification](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/qwen3#transformers.Qwen3ForSequenceClassification).
- [PEFT, modelos para classificação](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model).
- [OpenAI, endpoint Chat Completions](https://platform.openai.com/docs/api-reference/chat).
- [OpenAI, modelo gpt-4.1-mini e preços](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
- [PEFT, quantização e treinamento de adaptadores](https://huggingface.co/docs/peft/en/developer_guides/quantization).
- [Guo et al., On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
- [PyTorch, distribuições oficiais por versão](https://pytorch.org/get-started/previous-versions/).

## Apêndice, arquivos Python

Os links abaixo apontam para os scripts no repositório, copiados na seção 5.

### `decision_core.py`

Código: [decision_core.py](scripts/lab04-decisoes/decision_core.py).

### `prepare_abcd.py`

Código: [prepare_abcd.py](scripts/lab04-decisoes/prepare_abcd.py).

### `decision_lab.py`

Código: [decision_lab.py](scripts/lab04-decisoes/decision_lab.py).

### `evaluate_decisions.py`

Código: [evaluate_decisions.py](scripts/lab04-decisoes/evaluate_decisions.py).

### `serve_decisions.py`

Código: [serve_decisions.py](scripts/lab04-decisoes/serve_decisions.py).

Scripts auxiliares: [inspecionar_dados.py](scripts/lab04-decisoes/inspecionar_dados.py), [count_api_calls.py](scripts/lab04-decisoes/count_api_calls.py).

## Executar os comandos por script Bash

O arquivo [laboratorio-04-slm-decisoes-abcd.sh](laboratorio-04-slm-decisoes-abcd.sh) reúne os 20 blocos Bash deste roteiro em etapas numeradas. Liste as etapas e execute um intervalo:

```bash
bash ~/git/slm-lab/laboratorio-04-slm-decisoes-abcd.sh --list
bash ~/git/slm-lab/laboratorio-04-slm-decisoes-abcd.sh --steps 1-5
```

Ajuste o caminho do repositório se necessário. Execute as etapas seguintes em ordem conforme o roteiro; selecione as alternativas adequadas. Configuração de API e chamadas devem ficar na mesma execução. O servidor do laboratório 04 ocupa o terminal, e a chamada HTTP deve ser feita em outro terminal. Downloads manuais e caminhos de exemplo continuam exigindo os ajustes descritos no texto.

Cada execução grava stdout e stderr em `logs/laboratorio-04-slm-decisoes-abcd-DATA-HORA-PID.log` na pasta do laboratório. Os logs específicos dos comandos também são mantidos. Uma falha interrompe o script e registra o código de saída. O script usa a pasta do laboratório indicada neste roteiro; `LAB_WORK_DIR` permite escolher outra.
