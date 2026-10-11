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
