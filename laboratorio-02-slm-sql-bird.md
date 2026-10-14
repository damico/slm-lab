# Laboratório 2, SLM especialista em consultas SQL

**José Damico · Laboratório local · Versão 1.0 · 10/10/2026**

**Objetivo:** ajustar um Qwen3-1.7B para converter pergunta, esquema e evidência em SQL; executar as consultas geradas e comparar o resultado com o gabarito, com o modelo original e com um LLM maior.

## 1. Resultado que você construirá

Uma interface de linguagem natural para consultas de leitura sobre bancos. A especialização avaliada é **text-to-SQL em bancos cujos esquemas são fornecidos no prompt**. O treino ensina uma tarefa; não incorpora automaticamente os dados atuais do banco nos pesos.

O treino usa [birdsql/bird23-train-filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered), com 6.601 exemplos filtrados do BIRD. A avaliação final usa o split `mini_dev_sqlite` de [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev), com 500 perguntas e seus bancos. Uma parte dos bancos do conjunto de treino fica reservada para validação. O Mini-Dev é tratado como teste final deste laboratório; apesar do nome, não será usado para ajustar hiperparâmetros.

A página oficial do conjunto filtrado mostra ganhos de ajuste de um Qwen2.5-3B sobre a base. É evidência externa de utilidade do dataset, não uma previsão para nosso Qwen3-1.7B/QLoRA. O desempenho precisa ser medido.

Começaremos com **SQLite**, o caminho mais direto para executar os bancos do benchmark sem configurar um servidor. Sua preferência por PostgreSQL faz sentido para a integração futura, mas SQL aprendido/avaliado em SQLite não deve ser anunciado como validado em PostgreSQL. Ao final há uma orientação de expansão para o dialeto PostgreSQL.

```bash
mkdir -p "$HOME/slm-labs/02-sql"
cd "$HOME/slm-labs/02-sql"
```

Os scripts deste roteiro estão no repositório e podem ser utilizados sem executar o laboratório de ferramentas.

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

Os scripts estão em [scripts/lab02-sql](scripts/lab02-sql) no repositório. Dentro da pasta de trabalho do laboratório, copie-os antes de seguir:

`cp -n` copia os scripts da pasta específica deste laboratório no repositório para a pasta atual. Isso permite executar os comandos `python arquivo.py` e resolver os imports locais, mantendo dados e resultados isolados dos outros laboratórios. A opção `-n` preserva arquivos de destino que já existem.

```bash
cp -n ~/git/slm-lab/scripts/lab02-sql/*.py ./
```

O exemplo considera o repositório em `~/git/slm-lab`; ajuste o caminho se necessário. `cp -n` preserva scripts existentes. Para atualizar uma cópia antiga, compare-a com a versão do repositório antes de substituí-la.

O `lab_runtime.py` contém carregamento, treinamento e inferência. O `metrics.py` calcula métricas, intervalos de Wilson e diferenças pareadas. Os demais scripts são específicos do laboratório.

O modelo inicial é **Qwen/Qwen3-1.7B**, em modo sem raciocínio explícito (`enable_thinking=False`). O treino aplica a perda somente à resposta; tokens do prompt e padding recebem rótulo `-100`. A base fica quantizada em NF4 e congelada; os adaptadores LoRA são treinados nas camadas lineares. Isso é fine-tuning supervisionado com QLoRA, e não pré-treinamento do zero.

Na primeira preparação, os scripts registram os commits do modelo e datasets em `revisions.json`. Reexecuções reutilizam essas revisões. Preserve esse arquivo com os resultados.

## 4. Baixar os exemplos e os bancos

**`load_dataset(...)` baixa perguntas e SQLs, mas não substitui o download dos bancos.** Sem os arquivos `.sqlite`, não há esquema real nem avaliação por execução.

Planeje espaço: o BIRD trabalha com bancos grandes, na ordem de dezenas de GB. Como você manterá os ZIPs e as cópias extraídas, reserve aproximadamente **100 GB livres adicionais como margem inicial**, conferindo o tamanho efetivo dos pacotes antes de baixar. Isso é planejamento de disco, não uma medida feita neste ambiente.

O download de `train.zip` fornece os bancos SQLite de treino do BIRD. Eles são necessários para ler os esquemas usados nos prompts e executar SQLs durante a avaliação; perguntas e respostas SQL, sozinhas, não contêm os bancos. O arquivo é salvo em `raw/bird_train.zip` para extração posterior. O link oficial de treino, referenciado no card do dataset filtrado, é:

```bash
mkdir -p raw
curl -fL --retry 3 -C - \
  'https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip' \
  -o raw/bird_train.zip
```

O download separado do pacote Mini-Dev fornece os bancos SQLite dos casos de teste, necessários para executar as consultas previstas e as consultas de referência. Copiar o ZIP baixado pelo navegador para `raw/bird_mini.zip` dá ao extrator o nome e o local esperados pelos comandos abaixo.

Para o Mini-Dev, use **Download BIRD Mini-Dev Complete Package** na [página oficial](https://huggingface.co/datasets/birdsql/bird_mini_dev). O link apontado na consulta desta versão é [minidev_0703.zip no Google Drive](https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view). Faça o download pelo navegador e copie-o para `raw/bird_mini.zip` dentro desta pasta. Não salve a página HTML do Drive como se fosse um ZIP. Se o link tiver mudado, siga o link atual do card e registre a mudança.

O script de extração abaixo lida com ZIPs internos; os caminhos finais são encontrados recursivamente. Copie os scripts conforme a seção 3 e então execute:

Após extrair os bancos, `prepare_sql.py` baixa do Hugging Face as perguntas, evidências e SQLs de referência de `birdsql/bird23-train-filtered` e do split `mini_dev_sqlite` de `birdsql/bird_mini_dev`. Esses registros conectam cada pergunta ao banco correspondente e fornecem os gabaritos de treino e avaliação. O tokenizer do Qwen3-1.7B também é baixado para formatar os prompts e conferir o limite de tokens.

```bash
python extract_archives.py raw/bird_train.zip raw/bird_train
python extract_archives.py raw/bird_mini.zip raw/bird_mini
sha256sum raw/bird_train.zip raw/bird_mini.zip > raw/archive-sha256.txt
python prepare_sql.py --train-root raw/bird_train --test-root raw/bird_mini \
  --out data --max-seq 2048 --max-new 512
cat data/source_report.json
cat data/length_report.json
```

O extrator ignora `__MACOSX` e arquivos com prefixo `._`, que são metadados do macOS e podem ter extensão `.zip` sem serem arquivos ZIP. Os bancos e os ZIPs internos de dados continuam sendo extraídos; um ZIP de dados inválido ainda causa erro.

O extrator exige um destino vazio para evitar mistura de versões. Se interromper a extração, use outra pasta de destino e ajuste os argumentos de preparação, ou renomeie a pasta incompleta antes de tentar novamente. Não apague os arquivos originais que pretende preservar. Ao atualizar `extract_archives.py`, copie explicitamente a versão corrigida do repositório para a pasta do laboratório: a cópia inicial com `cp -n` preserva a versão antiga que já existe.

A preparação:

1. Resolve revisões dos dois datasets e do modelo.
2. Localiza os bancos de treino e Mini-Dev; interrompe se faltar algum ou houver duplicidade.
3. Extrai nomes, tipos, chaves primárias e relações de chaves estrangeiras.
4. Reserva aproximadamente 10% dos **bancos** de treino para validação; os demais servem ao treino.
5. Verifica que bancos do Mini-Dev não coincidem com os bancos do conjunto de treino.
6. Monta o prompt com esquema, pergunta e `evidence`, sem utilizar o SQL correto para escolher tabelas ou colunas.
7. Registra limites de contexto, exclusões e caminhos reais dos bancos.

O campo `evidence` é uma informação adicional fornecida pelo benchmark. Todos os modelos recebem a mesma evidência. No uso real essa informação pode não existir, e você precisaria de descrições ou catálogo de dados. Portanto, o resultado deve ser descrito como **text-to-SQL com esquema e evidência fornecidos**.

O esquema compacto evita parte do custo de contexto, mas ainda haverá bancos grandes que não cabem em 2.048 tokens. Esses casos são contabilizados, não truncados. Descrições completas de colunas e amostras de valores não são incluídas nesta primeira receita. Não compare seus números diretamente com sistemas que usam recuperação de valores, amostras, múltiplas tentativas ou correção via execução.

```bash
python inspecionar_dados.py
```

## 5. Arquivos Python necessários

Os cinco arquivos principais estão listados abaixo. `lab_runtime.py` e `metrics.py` são independentes das cópias do laboratório 1; execute todos os comandos a partir da pasta `02-sql`.

### lab_runtime.py

Código: [lab_runtime.py](scripts/lab02-sql/lab_runtime.py).

### metrics.py

Código: [metrics.py](scripts/lab02-sql/metrics.py).

### extract_archives.py

Código: [extract_archives.py](scripts/lab02-sql/extract_archives.py).

### prepare_sql.py

Código: [prepare_sql.py](scripts/lab02-sql/prepare_sql.py).

### score_sql.py

Código: [score_sql.py](scripts/lab02-sql/score_sql.py).

## 6. Verificar memória e treinar

Nos comandos `torchrun`, o separador `--` antes do nome do script encerra as opções do lançador. Assim, `--run` e os demais argumentos seguintes chegam ao script de treino sem serem interpretados como abreviações de opções do `torchrun`.

Na primeira execução do treino ou da inferência local, o runtime baixa do Hugging Face os pesos e a configuração do Qwen3-1.7B que ainda não estiverem no cache. Esses arquivos são a base pré-treinada sobre a qual o laboratório ajusta os adaptadores e, no laboratório 04, a camada de classificação. Inferências posteriores reutilizam o cache e carregam o adaptador da pasta local indicada; se algum arquivo do tokenizer ou da base estiver ausente, ele também é obtido do Hugging Face.

A configuração inicial é: **contexto total 2.048 tokens, reserva de até 512 tokens para saída, LoRA rank 16, learning rate 0,0001, microbatch 1, acumulação 8 e duas épocas**. É um ponto de partida, não uma garantia de encaixe em 8 GB nem uma configuração ótima demonstrada. O contexto inclui o catálogo/esquema, a pergunta e a resposta.

Primeiro, faça um teste de 10 passos em uma GPU. Ajuste o índice se a GPU 1 não estiver livre:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py train \
  --data data --run runs/smoke_1gpu \
  --max-steps 10 --accum 1 --eval-limit 32 \
  2>&1 | tee logs/smoke_1gpu.log
```

Depois, verifique o caminho de duas GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -- \
  lab_runtime.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-limit 32 \
  2>&1 | tee logs/smoke_2gpu.log
```

Os adaptadores desses testes curtos são apenas diagnósticos. O treinamento principal começa novamente do modelo original em outra pasta:

```bash
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

Para comparar com um modelo servido pela OpenAI, use `gpt-4.1-mini` no endpoint oficial de Chat Completions. O teste BIRD fornece até 500 perguntas; o script chama a API uma vez por linha elegível. A chave é lida sem eco no terminal e não deve ser salva no Markdown nem no Git. A API é cobrada conforme os tokens processados; veja a estimativa abaixo e confirme as [tarifas atuais do modelo](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

```bash
export LLM_URL='https://api.openai.com/v1/chat/completions'
export LLM_MODEL='gpt-4.1-mini'
read -r -s -p 'Chave da API OpenAI: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"max_tokens":512}'
```

O limite `max_tokens=512` é uma reserva para a resposta SQL; o script inclui essa reserva na preparação da janela, e o avaliador marca respostas truncadas como falha. Faça primeiro o piloto de uma pergunta e confira o retorno antes do teste completo.

Antes do piloto, conte as chamadas previstas. Depois faça uma chamada para confirmar compatibilidade e inspecionar o uso de tokens:

```bash
python count_api_calls.py --test
python lab_runtime.py predict --api --limit 1 \
  --input data/validation_demo.jsonl --output results/api_smoke.jsonl
```

Esse arquivo de uma chamada serve apenas para inspeção. Não o pontue contra os 50 casos como se fosse uma avaliação completa. Se houver `api_error`, corrija credenciais, endpoint ou parâmetros antes de continuar. Então:

```bash
python lab_runtime.py predict --api \
  --input data/test.jsonl --output results/test_llm.jsonl
python count_api_calls.py --predictions results/test_llm.jsonl
unset LLM_API_KEY
```

O BIRD Mini-Dev contém até 500 perguntas; o contador informa quantas são elegíveis depois do limite de contexto. Como referência de ritmo, a execução OpenAI do laboratório 04 processou cerca de 73 chamadas por minuto; 500 chamadas SQL levariam aproximadamente 7 minutos nesse ritmo, mas entradas SQL maiores ou a latência do serviço podem aumentar o tempo.

Com `gpt-4.1-mini` (tarifa consultada em 12/10/2026: US$ 0,40 por milhão de tokens de entrada e US$ 1,60 por milhão de saída), uma conta conservadora para 500 casos, até 2.048 tokens de entrada e até 512 de saída por caso, fica abaixo de **US$ 0,82**. O custo real tende a ser menor e é reportado pelo contador depois da execução. `--limit` permite um piloto; a comparação final deve utilizar o mesmo conjunto de IDs para todos. Para repetir, use outro arquivo de saída: o script impede sobrescrita acidental e não retoma inferência parcial automaticamente.

**Protocolo de comparação:** todos recebem a mesma informação no prompt e produzem uma resposta textual no contrato especificado. Este script não ativa o modo nativo de tool calling ou saídas estruturadas do provedor. Um LLM com suporte nativo pode obter outro resultado; essa configuração merece uma linha adicional na comparação de produção, com seu próprio cliente e adaptação de formato. Não apresente a condição de prompt uniforme como o melhor desempenho possível de cada plataforma.

A preparação marca `eligible=false` quando o prompt mais a reserva de saída excede o contexto escolhido. Esses casos continuam no teste, recebem `context_exceeded` em todas as condições e aparecem na cobertura. Assim, não se corta silenciosamente um esquema ou catálogo. A elegibilidade do teste não usa o tamanho da resposta correta. Os dados de treino/validação também excluem respostas que excedam o contexto; as contagens ficam nos relatórios.

## 8. Avaliar executando as consultas

```bash
python score_sql.py data/validation_demo.jsonl results/val_base.jsonl --out results/val_base
python score_sql.py data/validation_demo.jsonl results/val_tuned.jsonl --out results/val_tuned
```

Depois do treinamento/configuração final e das três inferências completas:

```bash
python score_sql.py data/test.jsonl results/test_base.jsonl --out results/base
python score_sql.py data/test.jsonl results/test_tuned.jsonl --out results/tuned
python score_sql.py data/test.jsonl results/test_llm.jsonl --out results/llm
python metrics.py results/base.cases.jsonl results/tuned.cases.jsonl
python metrics.py results/llm.cases.jsonl results/tuned.cases.jsonl
```

Cada consulta é executada com conexão `mode=ro`, `PRAGMA query_only=ON`, extensões desativadas, autorizador que permite leituras e funções SQL, limite de linhas e um limite de tempo acompanhado pelo SQLite. Comandos de escrita, `ATTACH`, alterações de PRAGMA e carregamento de extensão são recusados. Use os bancos públicos locais do laboratório; não aponte esse avaliador para um banco de produção. Esse isolamento é parte da implementação de um executor de consultas geradas, não uma validação de autorização empresarial.

O timeout padrão é 15 segundos por consulta. Ele usa o progress handler do SQLite, não um sandbox de processo com limite rígido de CPU/memória. Para um serviço exposto a usuários, coloque o executor em um processo/container isolado com limites do sistema. Para consultas legítimas mais lentas, defina antecipadamente outro `--timeout` e aplique o mesmo valor em todas as condições.

Métricas:

- **`accuracy_eligible` / EX-set local:** compara o conjunto de linhas retornadas pela consulta gerada com o conjunto do SQL de referência. O texto do SQL pode ser diferente.
- **`ex_bag`:** compara também a multiplicidade das linhas retornadas.
- **`execution_rate`:** consulta válida e executável, mesmo que responda à pergunta errada.
- **`by_difficulty`:** acerto por nível de dificuldade do Mini-Dev.
- **`gold_empty_n`:** referências que retornaram vazio, úteis para revisar acertos acidentais.

As métricas locais não verificam ordem das linhas e não aplicam tolerância numérica a diferenças de ponto flutuante. EX-set ignora duplicatas; EX-bag as preserva. Consultas diferentes podem coincidir por acaso em um único estado do banco. Uma resposta com ordenação incorreta pode passar aqui; revise casos com ordenação e resultados vazios e, para alegações formais, use testes adicionais e o avaliador oficial. A comparação de tuplas mantém a ordem das colunas.

Se a própria referência falhar ou exceder o limite, o caso é registrado como `reference_valid=false` e não entra no denominador válido. Investigue esses casos **independentemente do resultado dos modelos**. As três condições precisam ter o mesmo conjunto de referências válidas; o comparador pareado rejeita divergências. Uma consulta do modelo que falha diante de referência válida conta como erro.

**Este avaliador é uma medida local explícita, não uma implementação certificada do score oficial BIRD.** Para publicar uma posição ou comparar com o leaderboard, utilize a versão correspondente dos dados e o [avaliador oficial](https://github.com/bird-bench/mini_dev/tree/main/evaluation). Não misture os 500 casos SQLite do Mini-Dev clássico com outras extensões que incluem CRUD.

### Expansão posterior para PostgreSQL

O Mini-Dev também disponibiliza o split `mini_dev_pg` e material de construção dos bancos PostgreSQL. Faça essa expansão em outro experimento: importe o pacote oficial em um banco separado, use um usuário somente leitura, ajuste prompt/dialeto e troque o executor SQLite por um cliente PostgreSQL. Treinar apenas com SQL SQLite e avaliar PostgreSQL mede transferência entre dialetos, não um especialista PostgreSQL plenamente treinado.

Para integrar ao InfiniteStack, acrescente seleção autorizada de schemas, descrições de colunas, limite de execução e validação de consulta no backend. Uma demonstração de três perguntas de negócio contra um banco de exemplo é suficiente para mostrar o caminho de produto; ela permanece separada da avaliação quantitativa BIRD.

## 9. Interpretar os resultados e demonstrar valor

Execução concluída em **12/10/2026**, no equipamento com duas RTX 3070 de 8 GB. Os números abaixo foram lidos dos relatórios reais em `~/slm-labs/02-sql`; uma cópia dos resumos, metadados, revisões e hashes está em [resultados preservados](avaliador/docs/lab02-resultados-2026-10-12.json).

Foram preparados 5.262 exemplos de treino (750 excluídos por tamanho) e 588 de validação, com separação por bancos. Das 500 perguntas de teste, 449 cabem na janela de 2.048 tokens com reserva de 512 tokens de saída, cobertura de **89,80%**. Duas referências (`test:518` e `test:701`) excederam o timeout SQLite de 15 segundos nas três condições; restam **498 referências válidas e 447 casos elegíveis/válidos**. Isso indica uma limitação desta execução com timeout, não prova que esses SQLs estejam incorretos.

| Condição | Acertos / 447 | EX-set elegíveis | Wilson 95% | EX-set / 498 válidos | Latência p50 / p95 | Pico PyTorch alocado / reservado |
|---|---:|---:|---|---:|---|---|
| Qwen3-1.7B original | 65 | 14,54% | 11,58% a 18,11% | 13,05% | 2,50 s / 5,80 s | 1,45 / 2,35 GiB |
| Qwen3-1.7B ajustado | 114 | 25,50% | 21,68% a 29,74% | 22,89% | 5,49 s / 8,95 s | 1,56 / 2,39 GiB |
| OpenAI `gpt-4.1-mini` | 230 | 51,45% | 46,83% a 56,05% | 46,18% | 1,03 s / 1,54 s | não observável em API |

| Condição | Taxa executável / 447 | EX-bag / 447 | Simples (134) | Moderado (224) | Difícil (89) |
|---|---:|---:|---:|---:|---:|
| Original | 50,34% | 12,98% | 31,34% | 8,48% | 4,49% |
| Ajustado | 63,09% | 23,27% | 43,28% | 19,64% | 13,48% |
| `gpt-4.1-mini` | 97,76% | 47,65% | 67,91% | 45,98% | 40,45% |

A comparação pareada do ajuste contra o original mostra **+10,96 pontos percentuais**, bootstrap 95% de **+6,71 a +15,21 pp**: 72 casos melhoraram e 23 pioraram. Contra a API, o ajuste ficou **25,95 pp abaixo**, intervalo de **-30,87 a -21,03 pp**: venceu em 21 casos e perdeu em 137. O bootstrap é por pergunta e não controla a correlação entre perguntas do mesmo banco.

O fine-tuning melhorou o SLM neste recorte, mas não alcançou o LLM de referência nem reduziu a latência de geração: o especialista gerou respostas mais lentas nesta medição. Para a aula, mostre o ganho sobre o pequeno modelo original e as falhas que ainda exigem outro modelo ou revisão. Na demonstração de validação, houve 12/48 acertos no original e 24/48 no ajustado; esses 50 exemplos iniciais não substituem o teste completo.

### Treinamento e consumo medidos

O adaptador avaliado está em `runs/main_r16_lr1e4/final_adapter`: duas épocas, LoRA rank 16, learning rate `0.0001`, acumulação 8, microbatch 1 por GPU, seed 42 e avaliação limitada a 256 exemplos. Foram 658 passos, `train_loss=0.18087`, tempo do Trainer **1h17min42s** e tempo total de `/usr/bin/time` **1h19min02s**, com código de saída 0 e adaptador final gravado. O relatório registra pico alocado de **5,63 GiB no rank 0**. O monitor `nvidia-smi` registrou picos de **7.002 MiB na GPU 0** e **7.413 MiB na GPU 1**; memória de processos/contextos CUDA é diferente da memória alocada medida pelo PyTorch.

A integração das 2.371 amostras de potência por GPU durante o monitoramento estima **0,413 kWh nas duas GPUs** (0,229 + 0,184 kWh). Isso não inclui CPU, fonte, discos ou outros componentes. Para custo de energia do treino, multiplique esse consumo parcial pela tarifa local; amortização de hardware e operação ainda precisam de premissas declaradas.

Na inferência completa, as somas das latências registradas foram aproximadamente **30min00s no original**, **46min39s no ajustado** e **8min20s na API**. São somas de latências, não tempos de relógio completos com inicialização e pontuação. O original registrou 430 gerações `ok`, 19 `output_limit` e 51 `context_exceeded`; o ajustado, 444 `ok`, cinco `output_limit` e 51 `context_exceeded`; a API, 449 `ok` e 51 `context_exceeded`. As medianas e p95 da tabela usam apenas gerações `ok`, não apenas consultas SQL corretas.

### Custo da API observado

As **449 chamadas do teste** registraram **299.019 tokens de entrada e 27.584 de saída**. Pelas tarifas já utilizadas neste roteiro (US$ 0,40 e US$ 1,60 por milhão), o custo estimado do teste é **US$ 0,163742**, sem incluir o piloto de validação. O provedor pode aplicar descontos de cache; sem detalhamento de faturamento, esta conta usa a tarifa integral de entrada.

Isso equivale a **US$ 0,365 por 1.000 chamadas elegíveis com o mesmo perfil médio de tokens**, ou **US$ 0,327 por 1.000 perguntas apresentadas com a mesma cobertura** (incluindo as que não geram chamada). Não extrapole esses valores para perguntas de tamanho diferente. Não há custo local por 1.000 inferências completo medido; energia da inferência e amortização continuam dependentes de novas medições e premissas.

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

### Testar o adaptador no avaliador web

O [avaliador](avaliador/README.md) agora integra os Labs 02 e 04. Em um terminal, na máquina que contém os pesos e bancos, execute:

```bash
bash ~/git/slm-lab/avaliador/start-lab02.sh
```

O serviço usa a mesma revisão do modelo, tokenizer, template sem thinking e adaptador da execução medida; carrega os pesos uma vez e permanece ativo em `127.0.0.1:8082`. O script usa a GPU 1 por padrão, enquanto o Lab 04 usa a GPU 0. `LAB02_DIR`, `LAB02_ADAPTER`, `LAB02_GPU` e `LAB02_API_PORT` permitem ajustar pasta, artefato, GPU e porta. Não é necessário repetir treinamento, copiar datasets ou baixar bancos novamente.

Em outro terminal, inicie ou reinicie a interface:

```bash
bash ~/git/slm-lab/avaliador/run.sh
```

Abra `http://localhost:8080/labs/02`. Selecione banco e pergunta BIRD ou edite pergunta/evidence. A tela gera e executa SQL somente leitura, mostra as linhas retornadas, esquema, latência, tokens, etapas e confronto EX-set/EX-bag com o gabarito quando a entrada coincide com um exemplo. O gabarito nunca entra no prompt. Alterar a pergunta ou evidence remove sua associação com a referência.

As respostas original, ajustada e OpenAI do benchmark aparecem como **registradas**, sem novas chamadas. A opção de comparação OpenAI ao vivo usa `OPENAI_API_KEY` configurada no servidor Spring e faz uma chamada paga com as mesmas mensagens e até 512 tokens de saída. Ela também mostra tokens e custo estimado. `LAB02_API_URL` configura a conexão da interface à API SQL, independente de `LAB04_API_URL`.

O executor usa os bancos cadastrados no Mini-Dev, bloqueia escrita, extensões, `ATTACH` e PRAGMAs, limita consultas a 15 segundos pelo progress handler e 100.000 linhas, e exibe no máximo 100 linhas na interface. A restrição local segue os limites descritos na seção 8.

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

**Validação deste tutorial:** treinamento CUDA, inferências original/ajustada/API e avaliação completa foram executados pelo usuário e os relatórios conferidos em 12/10/2026. A integração web foi verificada com o adaptador treinado e bancos SQLite reais, além dos testes de contratos, contexto e restrições de execução. Os limites e denominadores das métricas aparecem na seção 9.

## 13. Referências primárias

Fontes consultadas em 12/10/2026; preserve revisões e hashes dos pacotes utilizados.

- [BIRD, conjunto de treino filtrado](https://huggingface.co/datasets/birdsql/bird23-train-filtered).
- [BIRD, Mini-Dev e splits por dialeto](https://huggingface.co/datasets/birdsql/bird_mini_dev).
- [BIRD, repositório Mini-Dev e avaliadores](https://github.com/bird-bench/mini_dev).
- [BIRD, artigo original](https://arxiv.org/abs/2305.03111).
- [Python, sqlite3, URI, autorizador e progress handler](https://docs.python.org/3/library/sqlite3.html).

- [Qwen3-1.7B, modelo, template e modo sem thinking](https://huggingface.co/Qwen/Qwen3-1.7B).
- [PEFT 0.17, quantização e QLoRA](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization).
- [Transformers 4.55.4, Trainer e TrainingArguments](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer).
- [PyTorch, instalações de versões anteriores](https://pytorch.org/get-started/previous-versions/).
- [bitsandbytes 0.47, requisitos de instalação](https://huggingface.co/docs/bitsandbytes/v0.47.0/en/installation).
- [Hugging Face Hub 0.34.4, autenticação e CLI](https://huggingface.co/docs/huggingface_hub/v0.34.4/en/guides/cli).
- [OpenAI, endpoint Chat Completions](https://platform.openai.com/docs/api-reference/chat).
- [OpenAI, modelo gpt-4.1-mini e preços](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

Scripts auxiliares: [inspecionar_dados.py](scripts/lab02-sql/inspecionar_dados.py), [listar_checkpoints.py](scripts/lab02-sql/listar_checkpoints.py), [preparar_demo.py](scripts/lab02-sql/preparar_demo.py), [count_api_calls.py](scripts/lab02-sql/count_api_calls.py).

## Executar os comandos por script Bash

O arquivo [laboratorio-02-slm-sql-bird.sh](laboratorio-02-slm-sql-bird.sh) reúne os 21 blocos Bash deste roteiro em etapas numeradas. Liste as etapas e execute um intervalo:

```bash
bash ~/git/slm-lab/laboratorio-02-slm-sql-bird.sh --list
bash ~/git/slm-lab/laboratorio-02-slm-sql-bird.sh --steps 1-5
```

Ajuste o caminho do repositório se necessário. Execute as etapas seguintes em ordem conforme o roteiro; selecione as alternativas adequadas. Configuração de API e chamadas devem ficar na mesma execução. As etapas 20 (API SQL) e 21 (interface Spring) ficam em primeiro plano: execute-as em terminais separados. Downloads manuais e caminhos de exemplo continuam exigindo os ajustes descritos no texto.

Cada execução grava stdout e stderr em `logs/laboratorio-02-slm-sql-bird-DATA-HORA-PID.log` na pasta do laboratório. Os logs específicos dos comandos também são mantidos. Uma falha interrompe o script e registra o código de saída. O script usa a pasta do laboratório indicada neste roteiro; `LAB_WORK_DIR` permite escolher outra.
