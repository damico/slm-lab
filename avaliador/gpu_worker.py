"""Inicializa as APIs originais com fila por GPU e liberação do cache após cada teste."""
import argparse
import contextlib
import fcntl
import functools
import importlib
import os
from pathlib import Path
import sys
import tempfile

SERVICES={
    '01':('lab01-ferramentas','serve_tools','Generator','__call__'),
    '02':('lab02-sql','serve_sql','SqlEngine','predict'),
    '03':('lab03-recibos','serve_receipts','Generator','__call__'),
    '04':('lab04-decisoes','serve_decisions','DecisionEngine','decide'),
    '05':('lab05-finqa','serve_finqa_web','FinqaLab','predict'),
}


@contextlib.contextmanager
def gpu_slot(devices=None):
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','') if devices is None else devices
    if not devices:
        yield;return
    # Cada serviço fica em uma GPU física; IDs iguais compartilham a fila.
    if not devices.isdigit(): raise ValueError('Use uma única GPU física por API do avaliador.')
    root=Path(tempfile.gettempdir())/f'slm-avaliador-gpu-{os.getuid()}'
    root.mkdir(mode=0o700,exist_ok=True)
    with open(root/f'{devices}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize();torch.cuda.empty_cache()
            fcntl.flock(lock,fcntl.LOCK_UN)


def serialized(method,skip_recorded=False):
    @functools.wraps(method)
    def call(*args,**kwargs):
        payload=args[1] if len(args)>1 else kwargs.get('payload')
        if skip_recorded and isinstance(payload,dict) and payload.get('mode','recorded')=='recorded' and 'context' in payload:
            return method(*args,**kwargs)
        with gpu_slot():return method(*args,**kwargs)
    return call


def main():
    p=argparse.ArgumentParser();p.add_argument('--lab',choices=SERVICES,required=True)
    args,remaining=p.parse_known_args();folder,module_name,class_name,method_name=SERVICES[args.lab]
    script_root=Path(__file__).resolve().parent.parent/'scripts'/folder
    sys.path.insert(0,str(script_root));module=importlib.import_module(module_name)
    cls=getattr(module,class_name)
    setattr(cls,method_name,serialized(getattr(cls,method_name),skip_recorded=args.lab=='05'))
    # FinQA carrega o aluno dentro de predict; SQL e ABCD o carregam ao iniciar.
    if args.lab in ('02','04'):cls.__init__=serialized(cls.__init__)
    sys.argv=[str(script_root/(module_name+'.py')),*remaining]
    module.main()

if __name__=='__main__':main()
