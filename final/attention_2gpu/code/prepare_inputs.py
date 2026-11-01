"""Controlled short text excerpts from immutable trace material,not DAG replay."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(cfg,out):
    from transformers import AutoTokenizer
    p=Path(cfg['corpus_path']);blob=p.read_bytes()
    assert hashlib.sha256(blob).hexdigest()==cfg['corpus_sha256']
    corpus=json.loads(blob);tok=AutoTokenizer.from_pretrained(cfg['model_path'],local_files_only=True)
    chosen=[];threads=set()
    for row in sorted(corpus['records'],key=lambda r:(r['input_tokens'],r['id'])):
        if row['thread'] in threads or row['input_tokens']<cfg['initial_tokens']:continue
        chosen.append(row);threads.add(row['thread'])
        if len(chosen)==8:break
    assert len(chosen)==8,'Need eight distinct recorded subagents'
    requests=[]
    marker='REPLACE_RECORDED_MATERIAL_HERE'
    for row in chosen:
        template=tok.apply_chat_template([
            dict(role='system',content='You are a coding subagent. Inspect the recorded repository material and report concrete findings. Tools are replayed by the benchmark.'),
            dict(role='user',content='Review this recorded task and repository excerpt. Continue the investigation in short steps.\n'+marker)],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        left,right=template.split(marker);a=tok.encode(left,add_special_tokens=False);z=tok.encode(right,add_special_tokens=False)
        # The source is data. Escape its original conversation delimiters before
        # embedding it in the controlled user/tool message.
        material=row['prompt_text'].replace('<|','[|').replace('|>','|]')
        ids=tok.encode(material,add_special_tokens=False)
        need=cfg['initial_tokens']-len(a)-len(z);assert len(ids)>=need>0
        initial=a+ids[-need:]+z;tools=[]
        for turn,n in enumerate(cfg['input_tokens_by_turn'][1:],1):
            lead=tok.encode('<|im_end|>\n<|im_start|>user\n[Replayed tool result '+str(turn)+']\n',add_special_tokens=False)
            tail=tok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
            count=n-1-len(lead)-len(tail);assert count>0
            start=max(0,len(ids)-(turn+1)*1024)
            body=ids[start:start+count];assert len(body)==count
            tools.append(lead+body+tail)
        requests.append(dict(source_id=row['id'],source_case=row['case'],thread=row['thread'],session_sha256=row['session_sha256'],
            initial_ids=initial,tool_ids=tools,scope='Explicitly shortened/reframed recorded text excerpts;not an unchanged trace prompt or task-quality benchmark.'))
    value=dict(status='CONTROLLED_MULTITURN_INPUTS_READY',requests=requests,config=cfg,
        pending_token='Each later prefill prepends the previous uncommitted last output token,then the recorded tool-material suffix. No sampled token is silently dropped or counted twice.',
        source_corpus_sha256=cfg['corpus_sha256'])
    out.write_text(json.dumps(value,ensure_ascii=False)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    v=prepare(json.loads(a.config.read_text()),a.out);print('PREPARED',len(v['requests']),flush=True)
