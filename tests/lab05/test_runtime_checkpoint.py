import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/lab05-finqa'))
from finqa_runtime import restore_lora_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_incomplete_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                restore_lora_checkpoint(None,folder)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('peft'),
                         'Requer o venv compartilhado com torch e peft')
    def test_real_adapter_restores_weights_and_preserves_training_parameters(self):
        import torch
        from peft import LoraConfig,get_peft_model

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.projection=torch.nn.Linear(4,4)
            def forward(self,x):
                return self.projection(x)

        model=get_peft_model(SmallModel(),LoraConfig(r=2,lora_alpha=4,target_modules=['projection']))
        with tempfile.TemporaryDirectory() as folder:
            with torch.no_grad():
                for name,param in model.named_parameters():
                    if 'lora_' in name:param.fill_(.25)
            expected={name:param.detach().clone() for name,param in model.named_parameters() if 'lora_' in name}
            model.save_pretrained(folder)
            with torch.no_grad():
                for name,param in model.named_parameters():
                    if name in expected:param.zero_()
            restore_lora_checkpoint(model,folder)
            for name,param in model.named_parameters():
                if name in expected:
                    self.assertTrue(torch.equal(param,expected[name]),name)
                    self.assertTrue(param.requires_grad,name)
                    self.assertEqual(param.device.type,'cpu')


if __name__=='__main__':unittest.main()
