# Preparação geral dos laboratórios de SLMs

Execute este guia uma vez antes dos laboratórios 01 a 04. Todos usam um único ambiente virtual em `~/slm-labs/.venv`, com as versões já fixadas nos roteiros. Cada laboratório mantém sua própria pasta de scripts, dados, logs e checkpoints: alguns scripts têm o mesmo nome e precisam ficar separados.

## 1. Conferir a máquina

Os roteiros consideram Ubuntu, Bash, Python 3.11 ou 3.12, duas RTX 3070 de 8 GB e aproximadamente 32 GB de RAM. Confira:

```bash
python3 --version
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi topo -m
df -h ~
```

O espaço necessário para dados e checkpoints está descrito em cada laboratório; um ambiente único reduz a duplicação de pacotes, mas os dados dos quatro experimentos continuam ocupando espaço. Execute os treinos em sequência quando usarem as duas GPUs.

Esta receita mantém o PyTorch 2.7.1 com runtime cu128 dos roteiros. A compatibilidade com o driver deve ser conferida na máquina antes do treino. A versão CUDA exibida por `nvidia-smi` descreve a capacidade do driver. Os wheels do PyTorch incluem seu runtime; estes laboratórios não exigem instalar um CUDA Toolkit separado.

## 2. Instalar ferramentas do sistema

Incluímos Tesseract e Poppler para o laboratório de recibos.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git curl unzip time tesseract-ocr tesseract-ocr-eng poppler-utils
```

## 3. Criar o ambiente compartilhado

Se você já estiver em um ambiente de laboratório, desative-o antes. Os ambientes antigos podem permanecer nas suas pastas; os comandos seguintes passam a usar o novo ambiente compartilhado.

```bash
if [ -n "${VIRTUAL_ENV:-}" ]; then deactivate; fi
mkdir -p ~/slm-labs
cd ~/slm-labs
python3 -m venv .venv
source ~/slm-labs/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
cat > requirements.txt <<'REQ'
transformers==4.55.4
peft==0.17.1
accelerate==1.10.1
bitsandbytes==0.47.0
datasets==4.0.0
huggingface-hub==0.34.4
Pillow==11.3.0
numpy==2.2.6
scipy==1.15.3
scikit-learn==1.7.1
joblib==1.5.1
REQ
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > requirements-resolved.txt
```

As versões acima são a união das dependências dos quatro roteiros, sem alterar seus pins. `requirements-resolved.txt` registra também as dependências transitivas instaladas. Execute os testes curtos de cada laboratório antes dos treinos completos.

## 4. Verificar imports e as duas GPUs

```bash
python - <<'PY'
import sys
from pathlib import Path
import torch
import transformers, peft, accelerate, bitsandbytes, datasets
import huggingface_hub, numpy, scipy, sklearn, joblib
from PIL import Image

expected = Path.home() / 'slm-labs' / '.venv'
assert Path(sys.prefix).resolve() == expected.resolve(), 'Ative o venv compartilhado'
assert sys.version_info[:2] in {(3, 11), (3, 12)}, 'Use Python 3.11 ou 3.12'
print('Python:', sys.executable)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA indisponivel'
assert torch.cuda.device_count() >= 2, 'As duas GPUs precisam estar visiveis'
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    p = torch.cuda.get_device_properties(i)
    x = torch.ones(8, device=f'cuda:{i}')
    assert (x + x).sum().item() == 16
    print(i, p.name, round(p.total_memory / 2**30, 2), 'GiB',
          'BF16:', torch.cuda.is_bf16_supported())
print('Imports e operacoes CUDA conferidos.')
PY
tesseract --version
pdftotext -v
```

Essa verificação cobre imports e uma operação simples em cada GPU. Os testes curtos dos roteiros conferem o carregamento quantizado e o treino dos modelos.

## 5. Usar o mesmo ambiente em qualquer laboratório

Em cada terminal novo, ative o ambiente pelo caminho completo. Mudar de pasta com `cd` mantém o ambiente ativo.

```bash
source ~/slm-labs/.venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_THREAD_LIMIT=1
```

Depois, entre na pasta do laboratório e siga seu roteiro:

| Laboratório | Roteiro |
|---|---|
| 01 — Ferramentas | [SLM e xLAM](laboratorio-01-slm-ferramentas-xlam.md) |
| 02 — SQL | [SLM e BIRD](laboratorio-02-slm-sql-bird.md) |
| 03 — Recibos | [Extração de recibos](laboratorio-03-slm-extracao-recibos.md) |
| 04 — Decisões | [Decisões com ABCD](laboratorio-04-slm-decisoes-abcd.md) |

Para continuar o laboratório 04 que você já iniciou:

```bash
cd ~/lab04-decisoes
source ~/slm-labs/.venv/bin/activate
cp ~/git/slm-lab/laboratorio-04-slm-decisoes-abcd.md ./laboratorio.md
cp ~/slm-labs/requirements.txt ./requirements.txt
cp ~/slm-labs/requirements-resolved.txt ./requirements-resolved.txt
mkdir -p raw runs results logs
export TOKENIZERS_PARALLELISM=false
```

O exemplo considera o repositório em `~/git/slm-lab`; ajuste esse caminho se necessário. Se os scripts já foram extraídos, continue na próxima etapa do roteiro. O extrator exige que eles ainda não existam para evitar sobrescrita.

Os caches de modelos e datasets do Hugging Face usam seus caminhos padrão no usuário e podem ser reaproveitados entre laboratórios. Os diretórios locais `raw`, `data` e `runs` continuam específicos de cada experimento.

Para sair do ambiente ao terminar:

```bash
deactivate
```
