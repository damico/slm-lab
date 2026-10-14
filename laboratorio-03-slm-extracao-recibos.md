# Laboratório 03, SLM especialista em extrair dados de recibos

**Aplicação:** transformar imagens, PDFs de uma página ou textos de recibos em JSON e CSV para organizar despesas e alimentar uma etapa de conferência de reembolsos.

**Equipamento-alvo:** duas RTX 3070 de **8 GB cada**. **Modelo:** Qwen3-1.7B com QLoRA. **Dataset pronto:** SROIE. **Entrega:** um adaptador treinado, um extrator local utilizável e uma comparação reproduzível com o modelo original e um LLM maior.

Este laboratório complementa o de SQL: um especialista transforma perguntas em consultas; o outro transforma documentos em dados estruturados. Você prepara e treina antes da apresentação de uma hora e demonstra os resultados ao vivo.

## 1. O que você vai conseguir usar

O comando final recebe um arquivo ou uma pasta e produz:

- `resultados.jsonl`: campos extraídos, saída original, status, arquivo de origem e hash.
- `despesas.csv`: uma linha por recibo, pronta para abrir em planilha ou importar em outro programa.
- `textos/`: texto efetivamente recebido pelo modelo, para investigar erros de OCR.

Os quatro campos são `company`, `date`, `address` e `total`. O total permanece uma **string**, evitando conversões monetárias silenciosas. A data mantém a representação encontrada. O sistema não aprova pagamentos nem determina se uma despesa é reembolsável.

Exemplo **ilustrativo**, não um resultado medido:

```json
{"company":"DEMO SUPPLIES LTD","date":"08/10/2026","address":"10 DEMO ROAD","total":"89.90"}
```

O OCR é executado pelo Tesseract. Você treina **o modelo de texto que interpreta o OCR**, não o reconhecedor de caracteres nem um modelo de visão. No PDF com texto selecionável, a aplicação usa `pdftotext`; em PDF digitalizado, renderiza a página e aplica OCR.

O escopo inicial é um recibo por arquivo e uma página por PDF. PDFs com várias páginas são rejeitados explicitamente. A interface é uma CLI local, com processamento em lote; não depende de uma interface web ou de ferramentas fictícias.

## 2. Dataset e o que o experimento demonstra

Usaremos [jsdnrs/ICDAR2019-SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE), distribuição do conjunto SROIE com imagens e os quatro campos anotados. Os metadados consultados em 10/10/2026 indicam **626 exemplos de treino e 361 de teste**. Um parágrafo antigo do card ainda menciona 347; o script registra o tamanho realmente baixado e fixa a revisão do repositório. A licença informada é CC BY 4.0; mantenha a atribuição ao compartilhar dados e materiais derivados.

São recibos em inglês, principalmente da Malásia. **Isso não valida NF-e, NFC-e, CNPJ, impostos ou recibos brasileiros.** A opção de OCR em português não substitui treino e avaliação nesse domínio. O resultado pode ser útil para documentos semelhantes aos avaliados; a extensão a outros documentos exige uma avaliação separada.

Não há trabalho manual de anotação neste roteiro. O script obtém os rótulos prontos e executa OCR nas imagens para construir as entradas. Ele reserva aproximadamente 10% do treino original para validação e preserva o teste original. Duplicatas exatas de imagem ou transcrição entre treino e teste saem do treino; duplicatas dentro do treino também são removidas antes da divisão.

Fornecedores podem aparecer em mais de um split. O relatório mostra o desempenho para nomes de fornecedor vistos e não vistos no treino, por correspondência textual normalizada. Isso não detecta todos os aliases de uma mesma empresa, nem garante independência de layouts. O conjunto também é público e pode estar no pré-treinamento dos modelos.

**Hipótese a testar:** especializar um modelo pequeno pode melhorar a extração nesse recorte e permitir operação local com qualidade suficiente. Não se presume que ele vencerá o LLM maior. Caso o maior mantenha vantagem relevante, esse é um resultado válido; você ainda mede quando custo, conectividade e qualidade justificam cada opção.

## 3. Ambiente e memória

Use Ubuntu, Bash, Python 3.11 ou 3.12 e, como ponto de partida, 32 GB de RAM e 25–40 GB livres em disco. O espaço inclui ambiente, cache e checkpoints. São estimativas de planejamento, não requisitos medidos neste computador.

**8 GB + 8 GB não viram automaticamente uma GPU de 16 GB.** O DDP replica o modelo quantizado em cada placa e distribui os exemplos. O modelo e as ativações de cada processo precisam caber em 8 GB. A segunda placa pode acelerar o treino, mas o ganho deve ser medido. A inferência deste laboratório usa uma placa.

Crie uma pasta nova. Salve uma cópia deste Markdown nela com o nome `laboratorio.md`:

A cópia do Markdown para `./laboratorio.md` deixa uma referência do roteiro na pasta do experimento. Ela é opcional para executar os scripts, que agora estão em arquivos `.py` separados.

```bash
mkdir -p lab03-recibos
cd lab03-recibos
cp ~/git/slm-lab/laboratorio-03-slm-extracao-recibos.md ./laboratorio.md
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
python3 --version
```

Prepare primeiro o ambiente compartilhado conforme [Preparação geral](preparacao-geral.md). Depois, ative-o abaixo. O runtime cu128 requer driver compatível; consulte a [matriz oficial do PyTorch](https://pytorch.org/get-started/previous-versions/) se a importação ou CUDA falhar. Não é necessário instalar um CUDA Toolkit separado para estes wheels.

O `cp` abaixo copia a receita única de dependências diretamente do repositório para a pasta do experimento. `python -m pip freeze` registra as versões efetivamente instaladas no venv ativo. Esses comandos não dependem de arquivos de requisitos em `~/slm-labs`, não criam outro venv e não instalam pacotes. Ajuste o caminho do repositório se necessário.

```bash
source ~/slm-labs/.venv/bin/activate
cp ~/git/slm-lab/requirements.txt ./requirements.txt
python -m pip freeze > requirements-resolved.txt
mkdir -p runs logs results raw
export TOKENIZERS_PARALLELISM=false
export OMP_THREAD_LIMIT=1
tesseract --version > logs/tesseract-version.txt 2>&1
pdftotext -v > logs/poppler-version.txt 2>&1
```

As versões foram fixadas para tornar o procedimento explícito; não representam uma recomendação de usar as versões mais recentes. O teste curto adiante verifica a combinação no seu hardware.

```bash
python ~/git/slm-lab/verificar_ambiente.py
```

## 4. Copiar os scripts do laboratório

Os scripts estão em [scripts/lab03-recibos](scripts/lab03-recibos). Copie-os para a pasta de trabalho com o comando abaixo. Ajuste o caminho do repositório se necessário. `cp -n` preserva arquivos existentes; compare cópias antigas antes de atualizá-las. O Markdown é apenas o roteiro e não precisa ser lido por um extrator.

`cp -n` copia os scripts da pasta específica deste laboratório no repositório para a pasta atual. Isso permite executar os comandos `python arquivo.py` e resolver os imports locais, mantendo dados e resultados isolados dos outros laboratórios. A opção `-n` preserva arquivos de destino que já existem.

```bash
cp -n ~/git/slm-lab/scripts/lab03-recibos/*.py ./
```

Os scripts usam apenas os pacotes instalados acima. O runtime é uma cópia independente, com avaliação mais frequente para este dataset pequeno; não copie por cima dos arquivos dos laboratórios anteriores.

## 5. Preparar os dados sem anotar exemplos

O comando `prepare_receipts.py` baixa do Hugging Face as imagens e os campos anotados do dataset `jsdnrs/ICDAR2019-SROIE`. As imagens são a entrada do OCR; as anotações são os gabaritos usados para treinar e avaliar a extração de empresa, endereço, data e total. O tokenizer do Qwen3-1.7B é baixado para formatar os exemplos e conferir o orçamento de tokens.

```bash
set -o pipefail
python prepare_receipts.py --out data --mode ocr --lang eng --psm 6 \
  --max-seq 2048 --max-new 512 2>&1 | tee logs/prepare.log
```

Na primeira execução, há download do tokenizer e dataset, seguido de OCR em CPU; os pesos do modelo serão baixados no primeiro treino ou inferência. O cache permite reutilizar o OCR em outra preparação com a mesma configuração. Alterar a versão do OCR, sua configuração ou o código compartilhado muda a chave do cache. Não edite os arquivos do cache manualmente.

O modo padrão treina e testa com **texto produzido pelo mesmo OCR usado pela aplicação**. Os rótulos supervisionados vêm do dataset. O modelo recebe somente o texto do recibo e as instruções: não recebe gabarito, nome do arquivo nem metadados do split.

Alguns registros de treino da fonte SROIE têm campos sem anotação. `X51005433522` tem `total` vazio; `X51005663280` tem `address=null`. O preparador exclui referências incompletas somente do treino e registra IDs, campos ausentes e anotações originais em `excluded_train_incomplete_references`, no relatório. Não inventa valores a partir do OCR nem transforma falta de anotação em alvo `null`. A ausência de anotação não prova que o campo esteja ausente na imagem. Os 361 registros de teste são preservados; se uma revisão futura trouxer um gabarito de teste incompleto, a preparação interrompe com uma mensagem específica.

Se você recebeu `Referencia ausente`, atualize o repositório e copie explicitamente `scripts/lab03-recibos/prepare_receipts.py` para sua pasta de trabalho. Essa cópia substitui o preparador corrigido; a cópia inicial com `cp -n` preserva versões antigas. Repita a etapa 5 do `.sh`. O diretório `data` da falha inicial está vazio e pode ser reutilizado; se já contiver arquivos, escolha outro `--out` para preservar os splits. O cache OCR em `raw/ocr_cache` é reutilizado.

Arquivos para conferir:

| Arquivo | O que verificar |
|---|---|
| `data/dataset_report.json` | Revisão, tamanhos originais, referências incompletas excluídas, duplicatas removidas, OCR vazio e fornecedores conhecidos |
| `data/length_report.json` | Quantos exemplos cabem no contexto e quantos foram excluídos do treino/validação |
| `data/train.jsonl` | Texto OCR em `text`, prompt em `messages`, resposta-alvo em `answer` |
| `data/validation.jsonl` | Casos para decisões sobre prompt, época e hiperparâmetros |
| `data/test.jsonl` | Teste preservado; use depois de congelar a receita |
| `data/demo_images/` | Dez imagens do teste, selecionadas por ID antes de conhecer os resultados |
| `revisions.json` | Commits exatos do modelo e dataset |

```bash
python inspecionar_dados.py
```

O limite inicial é 2.048 tokens no total, com reserva de 512 para a resposta. O treino e a validação rejeitam exemplos longos; no teste, os registros permanecem no arquivo e recebem `eligible=false`. Não há truncamento silencioso. A elegibilidade de teste depende somente do prompt, nunca do tamanho do gabarito. O resultado global conta recusas por contexto como falhas, e o relatório também mostra o resultado somente nos elegíveis.

Se quiser medir o efeito do OCR depois, prepare outro conjunto com `--out data_transcript --mode transcript`. Ele usa as transcrições fornecidas pelo dataset. Compare esse experimento separadamente e registre a diferença de entrada; não apresente seu resultado como a acurácia da aplicação com Tesseract. Não escolha o OCR olhando os acertos no teste.

## 6. Treinar nas duas RTX 3070

Nos comandos `torchrun`, o separador `--` antes do nome do script encerra as opções do lançador. Assim, `--run` e os demais argumentos seguintes chegam ao script de treino sem serem interpretados como abreviações de opções do `torchrun`.

Na primeira execução do treino ou da inferência local, o runtime baixa do Hugging Face os pesos e a configuração do Qwen3-1.7B que ainda não estiverem no cache. Esses arquivos são a base pré-treinada sobre a qual o laboratório ajusta os adaptadores e, no laboratório 04, a camada de classificação. Inferências posteriores reutilizam o cache e carregam o adaptador da pasta local indicada; se algum arquivo do tokenizer ou da base estiver ausente, ele também é obtido do Hugging Face.

Você fará ajuste supervisionado, não pré-treinamento do zero. A base fica congelada em NF4 com dupla quantização, e os adaptadores LoRA são treinados. A perda é calculada somente sobre os tokens da resposta; prompt e padding recebem `-100`. O runtime usa SDPA, gradient checkpointing e BF16 quando o hardware/runtime informar suporte, ou FP16 caso contrário.

Configuração inicial: LoRA rank 16, alpha 32, dropout 0,05, learning rate `0.0001`, três épocas, microbatch 1 e acumulação 8. Com duas GPUs, o batch efetivo é `1 × 8 × 2 = 16` exemplos. É um ponto de partida; o encaixe exato em 8 GB será verificado no teste curto.

Primeiro, teste uma GPU livre; depois, a comunicação entre as duas:

```bash
CUDA_VISIBLE_DEVICES=1 python lab_runtime.py train \
  --data data --run runs/smoke_1gpu --max-steps 10 --accum 1 \
  --eval-steps 5 --eval-limit 8 2>&1 | tee logs/smoke_1gpu.log

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 -- \
  lab_runtime.py train --data data --run runs/smoke_2gpu \
  --max-steps 20 --accum 8 --eval-steps 10 --eval-limit 16 \
  2>&1 | tee logs/smoke_2gpu.log
```

Os adaptadores dos testes curtos são diagnósticos. O treino principal começa novamente da base original:

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -l 2 > logs/gpu_train.csv &
LAB03_MONITOR_PID=$!
CUDA_VISIBLE_DEVICES=0,1 /usr/bin/time -v \
  torchrun --standalone --nproc-per-node=2 -- lab_runtime.py train \
  --data data --run runs/main --epochs 3 --accum 8 --rank 16 \
  --lr 0.0001 --eval-steps 20 --eval-limit 256 \
  2>&1 | tee logs/main_train.log
kill "$LAB03_MONITOR_PID"
```

A cada 20 atualizações, o Trainer avalia a perda na validação e salva checkpoint. Ao terminar, salva em `runs/main/final_adapter` o melhor checkpoint avaliado por `eval_loss`. São mantidos até dois checkpoints. A métrica de seleção é uma aproximação; a extração correta é medida na próxima seção.

O tempo depende do tamanho efetivo dos prompts, OCR, driver e comunicação entre GPUs. Estime a partir de vários passos após o aquecimento: aproximadamente `ceil(n_treino / 16) × épocas`, além da validação e gravação. Não há duração ou acurácia já medidas nas suas placas neste documento.

Se faltar memória: feche outros modelos; tente rank 8; se necessário, prepare outra pasta com `--max-seq 1536` e compare todas as condições com o mesmo novo limite. O microbatch já é 1. Reduzir a acumulação não reduz as ativações de um exemplo. Se usar uma GPU, `--accum 16` preserva o batch efetivo; mantenha o restante da receita.

Se o DDP falhar, confira os logs, driver e `nvidia-smi topo -m`. Não use `device_map="auto"` neste treino distribuído: cada processo deve permanecer na sua GPU. Para retomar, acrescente `--resume runs/main/checkpoint-N` usando um checkpoint realmente existente e os mesmos dados/configuração.

## 7. Validar antes de abrir o teste

```bash
CUDA_VISIBLE_DEVICES=0 python lab_runtime.py predict \
  --input data/validation.jsonl --output results/val_original.jsonl
CUDA_VISIBLE_DEVICES=0 python lab_runtime.py predict \
  --input data/validation.jsonl --output results/val_especialista.jsonl \
  --adapter runs/main/final_adapter
python score_receipts.py data/validation.jsonl results/val_original.jsonl --out results/val_original
python score_receipts.py data/validation.jsonl results/val_especialista.jsonl --out results/val_especialista
```

Inspecione os casos errados em `*.cases.jsonl` junto com `text`. Separe erros causados pelo OCR, troca de total por subtotal/troco, endereço incompleto e JSON inválido. Se o especialista piorou, teste uma mudança pequena, como uma época ou learning rate menor, sempre em nova pasta e avaliando somente na validação.

Depois, congele prompt, OCR, treino, normalização e configuração de geração. O dataset é pequeno: uma validação favorável não garante ganho no teste. Não aumente indefinidamente o número de tentativas e escolha apenas a melhor sem relatar isso.

## 8. Comparação objetiva: original, especialista e LLM maior

Execute os três com o **mesmo texto OCR**, as mesmas instruções e o mesmo contrato de saída. O original e o especialista usam a mesma base quantizada e geração sem amostragem. Essa é uma configuração de engenharia; não afirma medir a melhor pontuação possível de cada modelo. A recomendação geral de amostragem da família Qwen pode ser testada na validação com `--sample`, aplicada igualmente às condições locais.

```bash
CUDA_VISIBLE_DEVICES=0 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_original.jsonl
CUDA_VISIBLE_DEVICES=0 python lab_runtime.py predict \
  --input data/test.jsonl --output results/test_especialista.jsonl \
  --adapter runs/main/final_adapter
```

Para o LLM maior, uma opção de comparação é **Qwen3-32B em um endpoint externo compatível com Chat Completions**, usando modo sem raciocínio explícito. O modelo maior não será treinado nem hospedado nas suas duas placas neste roteiro. Você precisa de um endpoint acessível e de seu identificador de modelo. A chamada remota envia somente o texto dos exemplos; tem o custo e as condições do seu provedor.

Substitua os valores abaixo. `LLM_URL` deve ser a URL completa; o nome aceito pelo endpoint pode ser diferente do identificador no Hugging Face. O parâmetro `chat_template_kwargs` é adequado a servidores que o suportam, como configurações compatíveis de vLLM; não é universal. Adapte os parâmetros à API escolhida e confirme o modo de raciocínio na validação.

```bash
export LLM_URL='https://SEU_ENDPOINT/v1/chat/completions'
export LLM_MODEL='Qwen/Qwen3-32B'
read -r -s -p 'Chave do endpoint: ' LLM_API_KEY
export LLM_API_KEY
export LLM_EXTRA_JSON='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
python lab_runtime.py predict --api \
  --input data/validation.jsonl --output results/val_llm.jsonl --limit 3
```

Confira se os três pedidos de validação retornam JSON completo e se o endpoint não está gastando a reserva de saída em raciocínio. Para APIs com outra convenção de parâmetros, ajuste `LLM_EXTRA_JSON`. O cliente aceita `max_tokens` ou `max_completion_tokens`; por padrão usa `max_tokens=512`. Uma comparação com raciocínio habilitado deve ser identificada como outra condição, com seu orçamento registrado.

```bash
python lab_runtime.py predict --api \
  --input data/test.jsonl --output results/test_llm.jsonl
python score_receipts.py data/test.jsonl results/test_original.jsonl --out results/original
python score_receipts.py data/test.jsonl results/test_especialista.jsonl --out results/especialista
python score_receipts.py data/test.jsonl results/test_llm.jsonl --out results/llm
python metrics.py results/original.cases.jsonl results/especialista.cases.jsonl
python metrics.py results/llm.cases.jsonl results/especialista.cases.jsonl
```

Não use arquivos produzidos com `--limit` para reportar teste completo. Saídas ausentes contam como falhas. O runtime não sobrescreve arquivos de predição existentes: use outro nome para outra execução. O comparador pareado verifica se IDs e elegibilidade são iguais.

### Métricas e interpretação

A principal é **percentual de documentos com os quatro campos corretos**. A comparação normaliza Unicode, maiúsculas/minúsculas e espaços. Mantém pontuação e representação de valores: `89.9` e `89.90`, ou datas em formatos diferentes, não são automaticamente equivalentes. A regra é idêntica para todos; defina mudanças na validação.

O avaliador exige JSON puro, exatamente quatro chaves e valores string ou `null`. Rejeita texto extra, blocos Markdown, chaves repetidas e tipos incorretos. `null` é uma opção do contrato da aplicação, mas sua capacidade de abstenção em documentos sem campos não é demonstrada por este teste.

| Resultado | Origem | Como interpretar |
|---|---|---|
| `accuracy_all_valid` | `*.summary.json` | Quatro campos corretos, contando o teste inteiro e falhas por contexto |
| `accuracy_eligible` | `*.summary.json` | Quatro campos corretos apenas nos documentos dentro do contexto |
| `field_accuracy` | `*.summary.json` | Acerto individual de empresa, data, endereço e total |
| `schema_valid_rate` | `*.summary.json` | Saída parseável; não prova conteúdo correto |
| `merchant_groups` | `*.summary.json` | Resultado por nome de fornecedor visto/não visto |
| `p50_s`, `p95_s` | `*.summary.json` | Latência de geração das respostas com status `ok`; sem OCR/carga |
| VRAM, configuração | `*.meta.json`, `run_report.json`, logs | Memória alocada pelo PyTorch e uso observado pelo `nvidia-smi` |
| Tokens da API | `*.summary.json` | Base para calcular custo com a tarifa real do endpoint |

O bootstrap pareado reporta a diferença B − A, com intervalo de 95%. Se o intervalo incluir zero, não trate o ganho como estabelecido por esse teste. Os intervalos são aproximados e tratam documentos como unidades independentes; recibos do mesmo fornecedor podem ser correlacionados.

Na comparação de custo, reporte custo de API observado, tempo/energia local, amortização do treinamento e volume esperado. Rodar localmente não significa custo zero. Na comparação de latência, o remoto inclui rede; o cronômetro local começa após preparar a entrada na GPU. Mostre o número de sucessos junto ao p95: latência só das respostas concluídas pode esconder falhas.

Preencha esta tabela **depois** das execuções; não há números de resultado inventados no tutorial:

| Modelo | Documentos corretos/total | Total correto (%) | JSON válido (%) | p50/p95 geração | Memória ou custo medido |
|---|---|---|---|---|---|
| Qwen3-1.7B original | a medir | a medir | a medir | a medir | a medir |
| Qwen3-1.7B + adaptador | a medir | a medir | a medir | a medir | a medir |
| LLM maior, versão registrada | a medir | a medir | a medir | a medir | a medir |

## 9. Usar a aplicação de verdade

As dez imagens em `demo_images` já estão prontas para uma demonstração. Nenhum acerto foi usado para escolhê-las. Rode a pasta inteira:

```bash
CUDA_VISIBLE_DEVICES=0 python extract_receipts.py data/demo_images \
  --adapter runs/main/final_adapter --out saida-demo
```

Para seu próprio recibo, passe o caminho real do arquivo:

```bash
CUDA_VISIBLE_DEVICES=0 python extract_receipts.py /caminho/recibo.png \
  --adapter runs/main/final_adapter --out saida-recibo
CUDA_VISIBLE_DEVICES=0 python extract_receipts.py /caminho/recibo.pdf \
  --adapter runs/main/final_adapter --out saida-pdf
```

Cada execução requer uma pasta `--out` nova. O modelo é carregado uma única vez para o lote. Para comparar a aplicação com o modelo original, omita `--adapter` e use outra saída. O tempo por documento da aplicação inclui extração de texto/OCR, tokenização e geração, excluindo o carregamento do modelo; o primeiro documento ainda pode sofrer aquecimento. Não misture esses tempos com os do benchmark de geração.

Abra o CSV e confira o total de cada documento junto da imagem. O status `schema_valid` significa somente que a estrutura foi validada; **não é aprovação da despesa nem garantia de acerto**. O JSON original permanece disponível; o CSV protege células que poderiam ser interpretadas como fórmulas pela planilha. O CSV não faz soma, conversão cambial nem inferência da moeda.

A aplicação preserva saída bruta e texto OCR. Se o PDF tiver camada de texto ruim, acrescente `--force-ocr`. Para PDFs com várias páginas, separe-os primeiro, mantendo um recibo por arquivo; documentos em que um único recibo ocupa várias páginas estão fora deste roteiro.

Você também pode testar a integração sem imagem, com um texto sintético claramente identificado:

```bash
cat > exemplo_sintetico.txt <<'TXT'
DEMO SUPPLIES LTD
10 DEMO ROAD
DATE 08/10/2026
SUBTOTAL 84.81
TAX 5.09
TOTAL RM 89.90
CASH 100.00
CHANGE 10.10
TXT
CUDA_VISIBLE_DEVICES=0 python extract_receipts.py exemplo_sintetico.txt \
  --adapter runs/main/final_adapter --out saida-sintetico
```

O alvo esperado desse exemplo é o JSON ilustrativo da seção 1. Ele verifica integração e a distinção entre total/subtotal/troco; um acerto isolado não é evidência de qualidade geral. Para a apresentação, mostre casos corretos e erros representativos já analisados, junto da pontuação do conjunto inteiro.

## 10. O que torna esse caso convincente para engenheiros

A aplicação resolve uma transformação concreta e frequente: documento sem estrutura → registro importável. O experimento isola o efeito do fine-tuning usando a mesma base e o mesmo OCR, registra limites de hardware e permite checar a resposta contra rótulos sem usar um LLM como juiz.

O argumento forte para especialização é demonstrar uma relação útil entre **qualidade, recursos necessários e controle do processamento**. Se o especialista alcançar a qualidade requerida pelo seu processo numa GPU disponível, há valor mesmo que o LLM maior tenha alguma vantagem de acurácia. Defina esse requisito antes do teste; para valores monetários, mostre separadamente o acerto de `total` e a necessidade de conferência.

Se o modelo original já for suficiente, reporte que o ajuste não se justificou. Se regras simples resolverem os recibos de um único layout, vale medir essa alternativa também. O propósito é ensinar a escolher uma solução sustentada pelos resultados, não demonstrar superioridade universal de SLMs.

Para juntar SQL e recibos na apresentação de 60 minutos:

| Minutos | Conteúdo |
|---|---|
| 0–6 | Duas aplicações prontas e quais problemas resolvem |
| 6–16 | QLoRA, dados supervisionados, splits e limitações de 2 × 8 GB |
| 16–28 | SQL: consulta executada, comparação e erros |
| 28–40 | Recibos: imagem → CSV, comparação e erros |
| 40–50 | Evidências: qualidade, memória, tempo e escolha entre SLM/LLM |
| 50–60 | Perguntas e demonstração de um caso novo |

Faça downloads, OCR, treinamento e benchmark antes. Leve os resultados, adaptadores, imagens, CSVs e logs. Na hora, execute inferência; não dependa de finalizar o treinamento ou acessar uma API para concluir a explicação.

## 11. Verificação e referências

Este material foi verificado quanto à sintaxe dos scripts/blocos de terminal, parsing e métricas em casos controlados, e caminhos reais de OCR de imagem, extração de texto de PDF e OCR de PDF. O ambiente de autoria não dispõe do treinamento CUDA deste laboratório: **treino, consumo de VRAM e qualidade do modelo precisam ser executados e medidos no seu lab**.

Fontes primárias consultadas em 10/10/2026:

- [Huang et al., ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction](https://arxiv.org/abs/2103.10213).
- [Dataset e metadados SROIE da distribuição utilizada](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE).
- [Qwen3-1.7B, model card e modo sem thinking](https://huggingface.co/Qwen/Qwen3-1.7B).
- [Qwen3-32B, modelo geral de comparação](https://huggingface.co/Qwen/Qwen3-32B).
- [PEFT, treinamento com quantização](https://huggingface.co/docs/peft/en/developer_guides/quantization).
- [Transformers 4.55.4, Trainer](https://huggingface.co/docs/transformers/v4.55.4/en/trainer).
- [PyTorch, versões anteriores e wheels CUDA](https://pytorch.org/get-started/previous-versions/).
- [Tesseract, linha de comando](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).

## Apêndice, arquivos Python

Os links abaixo apontam para os arquivos Python copiados na seção 4. Todos os comandos do roteiro assumem a pasta `lab03-recibos` como diretório atual.

### `lab_runtime.py`

Código: [lab_runtime.py](scripts/lab03-recibos/lab_runtime.py).

### `metrics.py`

Código: [metrics.py](scripts/lab03-recibos/metrics.py).

### `receipt_common.py`

Código: [receipt_common.py](scripts/lab03-recibos/receipt_common.py).

### `prepare_receipts.py`

Código: [prepare_receipts.py](scripts/lab03-recibos/prepare_receipts.py).

### `score_receipts.py`

Código: [score_receipts.py](scripts/lab03-recibos/score_receipts.py).

### `extract_receipts.py`

Código: [extract_receipts.py](scripts/lab03-recibos/extract_receipts.py).

Scripts auxiliares: [inspecionar_dados.py](scripts/lab03-recibos/inspecionar_dados.py).

## Executar os comandos por script Bash

O arquivo [laboratorio-03-slm-extracao-recibos.sh](laboratorio-03-slm-extracao-recibos.sh) reúne os 15 blocos Bash deste roteiro em etapas numeradas. Liste as etapas e execute um intervalo:

```bash
bash ~/git/slm-lab/laboratorio-03-slm-extracao-recibos.sh --list
bash ~/git/slm-lab/laboratorio-03-slm-extracao-recibos.sh --steps 1-5
```

Ajuste o caminho do repositório se necessário. Execute as etapas seguintes em ordem conforme o roteiro; selecione as alternativas adequadas. Configuração de API e chamadas devem ficar na mesma execução. O servidor do laboratório 04 ocupa o terminal, e a chamada HTTP deve ser feita em outro terminal. Downloads manuais e caminhos de exemplo continuam exigindo os ajustes descritos no texto.

Cada execução grava stdout e stderr em `logs/laboratorio-03-slm-extracao-recibos-DATA-HORA-PID.log` na pasta do laboratório. Os logs específicos dos comandos também são mantidos. Uma falha interrompe o script e registra o código de saída. O script usa a pasta do laboratório indicada neste roteiro; `LAB_WORK_DIR` permite escolher outra.
