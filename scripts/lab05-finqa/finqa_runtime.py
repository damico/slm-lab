"""Inferência do professor separado; aluno QLoRA com DDP explícito."""
import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from finqa_core import STUDENT, TEACHER, read_json, read_rows, save_json, write_rows, file_hash, digest


def tokenizer(model):
    from transformers import AutoTokenizer
    lock=read_json('revisions.json')
    tok=AutoTokenizer.from_pretrained(model,revision=lock[model])
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    return tok


def prompt_ids(tok, messages):
    return tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)


def load_model(model, adapter=None, training=False, shard=False, gpu_memory='5GiB'):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available(): raise RuntimeError('CUDA indisponível')
    rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank)
    bf16=torch.cuda.is_bf16_supported();dtype=torch.bfloat16 if bf16 else torch.float16
    if shard and (training or int(os.environ.get('WORLD_SIZE','1'))>1): raise ValueError('Sharding só para inferência em um processo')
    extra={'max_memory':{i:gpu_memory for i in range(torch.cuda.device_count())}} if shard else {}
    net=AutoModelForCausalLM.from_pretrained(model,revision=read_json('revisions.json')[model],
        torch_dtype=dtype,device_map='balanced' if shard else {'':rank},attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype),**extra)
    if any(str(device) in ('cpu','disk') for device in getattr(net,'hf_device_map',{}).values()):
        raise RuntimeError('Mapa incluiu CPU/disco. Reduza janela ou ajuste --gpu-memory; não medir como inferência só em GPU')
    net.config.use_cache=not training
    if adapter:
        config=read_json(Path(adapter,'adapter_config.json'))
        if config['base_model_name_or_path']!=model: raise ValueError('Adaptador de outra base')
        report_path=Path(adapter).parent/'run_report.json'
        if report_path.exists() and read_json(report_path)['config']['models'][model]!=read_json('revisions.json')[model]:
            raise ValueError('Revisão da base difere do treino do adaptador')
        net=PeftModel.from_pretrained(net,adapter,is_trainable=False)
    return net,bf16


class Generator:
    def __init__(self, model, adapter=None, shard=False, gpu_memory='5GiB'):
        self.tok=tokenizer(model);self.model,_=load_model(model,adapter,shard=shard,gpu_memory=gpu_memory)
        self.model.eval()
        for key in ['temperature','top_p','top_k']:
            setattr(self.model.generation_config,key,None)

    def __call__(self, messages, max_new):
        import torch
        ids=prompt_ids(self.tok,messages)
        device=self.model.get_input_embeddings().weight.device
        tensor=torch.tensor([ids],device=device)
        synchronize();start=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(tensor,attention_mask=torch.ones_like(tensor),max_new_tokens=max_new,
                do_sample=False,pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
        synchronize();seconds=time.perf_counter()-start
        tokens=output[0,len(ids):].tolist()
        return {'text':self.tok.decode(tokens,skip_special_tokens=True).strip(),
            'status':'output_limit' if len(tokens)>=max_new and tokens[-1]!=self.tok.eos_token_id else 'ok',
            'input_tokens':len(ids),'output_tokens':len(tokens),'latency_s':seconds}


def synchronize():
    import torch
    for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)


def predict(args):
    import torch
    from transformers import set_seed
    if int(os.environ.get('WORLD_SIZE','1'))>1:raise ValueError('Inferência: use python, não torchrun')
    if args.model==TEACHER and args.adapter:raise ValueError('Professor não usa adaptador neste experimento')
    rows=read_rows(args.input)
    if args.limit:rows=rows[:args.limit]
    out=Path(args.output);meta=out.with_suffix('.meta.json')
    config=read_json(Path(args.data,'config.json'))
    if config['models'][args.model]!=read_json('revisions.json')[args.model]:raise ValueError('Revisão difere da preparação')
    signature={'input_sha256':file_hash(args.input),'model':args.model,
        'revision':read_json('revisions.json')[args.model],'limit':args.limit,'seed':args.seed,
        'max_seq':config['max_seq'],'max_new':config['max_new'],
        'adapter_sha256':file_hash(Path(args.adapter,'adapter_model.safetensors')) if args.adapter else None}
    done=set()
    if out.exists():
        if not args.resume or not meta.exists() or read_json(meta)['signature']!=signature:
            raise FileExistsError('Use outro --output ou --resume com a mesma assinatura')
        completed=read_rows(out);done={r['id'] for r in completed}
        if len(done)!=len(completed) or not done<={r['id'] for r in rows}:raise ValueError('IDs inválidos na retomada')
    elif args.resume:raise FileNotFoundError('Saída para retomar não existe')
    out.parent.mkdir(parents=True,exist_ok=True)
    pending=[r for r in rows if r['id'] not in done]
    if not pending:return
    report={'signature':signature,'arguments':vars(args),'prompt':'mesmas mensagens preparadas; sem gold',
            'started':time.time()};save_json(meta,report)
    set_seed(args.seed);generator=Generator(args.model,args.adapter,args.shard,args.gpu_memory)
    report['device_map']={str(k):str(v) for k,v in getattr(generator.model,'hf_device_map',{}).items()}
    save_json(meta,report)
    warm=next((r for r in pending if r['eligible']),None)
    if warm:generator(warm['messages'],8)
    for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i)
    with out.open('a' if done else 'w',encoding='utf-8') as stream:
        for row in pending:
            prediction={'id':row['id'],'model':args.model,'status':'context_exceeded','text':''}
            if row['eligible']:
                n=len(prompt_ids(generator.tok,row['messages']))
                if n+config['max_new']<=config['max_seq']:prediction.update(generator(row['messages'],config['max_new']))
            stream.write(json.dumps(prediction,ensure_ascii=False)+'\n');stream.flush()
            print(row['id'],prediction['status'],flush=True)
    report.update(completed=len(read_rows(out)),finished=time.time(),
        gpu_peak={str(i):{'allocated_gib':torch.cuda.max_memory_allocated(i)/2**30,
                           'reserved_gib':torch.cuda.max_memory_reserved(i)/2**30}
                  for i in range(torch.cuda.device_count())})
    save_json(meta,report)


def restore_lora_checkpoint(model, checkpoint):
    """Read the small adapter on CPU, then copy into this rank's existing parameters."""
    checkpoint=Path(checkpoint)
    if not (checkpoint/'adapter_model.safetensors').is_file():
        raise FileNotFoundError('Checkpoint LoRA local incompleto: '+str(checkpoint))
    return model.load_adapter(str(checkpoint),model.active_adapter,
                              torch_device='cpu',is_trainable=True)


def train(args):
    import torch
    from datasets import Dataset
    from transformers import Trainer,TrainingArguments,DataCollatorForSeq2Seq,set_seed
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    class SynchronizedTrainer(Trainer):
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            restore_lora_checkpoint(self.model if model is None else model,resume_from_checkpoint)

        def _load_best_model(self):
            restore_lora_checkpoint(self.model,self.state.best_model_checkpoint)
            print(f'DDP: rank local {os.environ.get("LOCAL_RANK","0")}, melhor adaptador recarregado via CPU.',flush=True)

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            # Rank 1 can inspect the checkpoint directory before rank 0 creates it.
            # Identical selection state keeps load_best_model_at_end collectives aligned.
            if torch.distributed.is_initialized():
                selection=[(self.state.best_model_checkpoint,self.state.best_metric,
                            self.state.best_global_step) if self.is_world_process_zero() else None]
                torch.distributed.broadcast_object_list(selection,src=0)
                (self.state.best_model_checkpoint,self.state.best_metric,
                 self.state.best_global_step)=selection[0]
    if not torch.cuda.is_available():raise RuntimeError('CUDA indisponível')
    rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank)
    distributed=int(os.environ.get('WORLD_SIZE','1'))>1
    if distributed:
        torch.distributed.init_process_group(backend='nccl',device_id=torch.device('cuda',rank),timeout=timedelta(minutes=5))
    try:
        set_seed(args.seed);tok=tokenizer(STUDENT);config=read_json(Path(args.data,'config.json'))
        if config['models'][STUDENT]!=read_json('revisions.json')[STUDENT]:raise ValueError('Revisão do aluno difere da preparação')
        parts={};counts={}
        for split in ['train','validation']:
            rows=read_rows(Path(args.data,split+'.jsonl'))
            if split=='validation' and args.eval_limit:rows=sorted(rows,key=lambda r:digest(r['id']))[:args.eval_limit]
            encoded=[]
            for row in rows:
                prefix=prompt_ids(tok,row['messages']);answer=tok.encode(row['answer'],add_special_tokens=False)+[tok.eos_token_id]
                if len(prefix)+len(answer)>config['max_seq']:raise ValueError('Alvo ultrapassou contexto: '+row['id'])
                encoded.append({'input_ids':prefix+answer,'attention_mask':[1]*(len(prefix)+len(answer)),
                                'labels':[-100]*len(prefix)+answer})
            if not encoded:raise ValueError('Split vazio: '+split)
            counts[split]={'examples':len(rows),'target_tokens':sum(sum(v!=-100 for v in r['labels']) for r in encoded)}
            parts[split]=Dataset.from_list(encoded)
        net,bf16=load_model(STUDENT,training=True)
        net=prepare_model_for_kbit_training(net,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        net=get_peft_model(net,LoraConfig(task_type='CAUSAL_LM',r=args.rank,lora_alpha=2*args.rank,
            lora_dropout=.05,target_modules='all-linear',bias='none'))
        net.print_trainable_parameters()
        options=TrainingArguments(output_dir=args.run,num_train_epochs=args.epochs,max_steps=args.max_steps,
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,warmup_ratio=.03,lr_scheduler_type='cosine',weight_decay=.01,max_grad_norm=1,
            optim='adamw_torch',bf16=bf16,fp16=not bf16,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},ddp_find_unused_parameters=False,ddp_broadcast_buffers=False,
            eval_strategy='steps',eval_steps=args.eval_steps,save_strategy='steps',save_steps=args.eval_steps,
            save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
            prediction_loss_only=True,logging_steps=10,report_to='none',seed=args.seed,data_seed=args.seed,
            dataloader_num_workers=0,label_names=['labels'])
        trainer=SynchronizedTrainer(model=net,args=options,train_dataset=parts['train'],eval_dataset=parts['validation'],
            processing_class=tok,data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100,pad_to_multiple_of=8))
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        result=trainer.train(resume_from_checkpoint=args.resume or None)
        print(f'DDP: rank local {rank}, Trainer concluído; melhor checkpoint: {trainer.state.best_model_checkpoint}',flush=True)
        if distributed:torch.distributed.barrier(device_ids=[rank])
        if trainer.is_world_process_zero():
            final=Path(args.run,'final_adapter');trainer.model.save_pretrained(final);tok.save_pretrained(final)
            save_json(Path(args.run,'run_report.json'),{**result.metrics,'seconds':time.perf_counter()-start,
                'rank0_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'world_size':int(os.environ.get('WORLD_SIZE','1')),
                'counts':counts,'arguments':vars(args),'config':config,
                'best_model_checkpoint':trainer.state.best_model_checkpoint,'best_eval_loss':trainer.state.best_metric,
                'data_sha256':{split:file_hash(Path(args.data,split+'.jsonl')) for split in parts}})
            print('Adaptador final e relatório gravados.',flush=True)
        if distributed:torch.distributed.barrier(device_ids=[rank])
    finally:
        if distributed and torch.distributed.is_initialized():torch.distributed.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    pred=sub.add_parser('predict');pred.add_argument('--data',default='data');pred.add_argument('--input',required=True)
    pred.add_argument('--output',required=True);pred.add_argument('--model',choices=[STUDENT,TEACHER],default=STUDENT)
    pred.add_argument('--adapter');pred.add_argument('--shard',action='store_true');pred.add_argument('--gpu-memory',default='5GiB')
    pred.add_argument('--limit',type=int,default=0);pred.add_argument('--resume',action='store_true');pred.add_argument('--seed',type=int,default=42)
    tr=sub.add_parser('train');tr.add_argument('--data',required=True);tr.add_argument('--run',required=True)
    tr.add_argument('--epochs',type=float,default=2);tr.add_argument('--max-steps',type=int,default=-1)
    tr.add_argument('--accum',type=int,default=8);tr.add_argument('--rank',type=int,default=16);tr.add_argument('--lr',type=float,default=1e-4)
    tr.add_argument('--eval-limit',type=int,default=128);tr.add_argument('--eval-steps',type=int,default=100)
    tr.add_argument('--seed',type=int,default=42);tr.add_argument('--resume')
    args=parser.parse_args()
    if hasattr(args,'limit') and args.limit<0:raise ValueError('Limit deve ser não negativo')
    train(args) if args.command=='train' else predict(args)


if __name__=='__main__':main()
