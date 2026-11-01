"""CPU regression of active/unused/in-flight page accounting; no GPU or oracle."""
import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
import torch


def main():
    tree=ast.parse((Path(__file__).parent/'backend.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReplayBackend')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace=dict(torch=torch,Counter=Counter);exec(compile(module,'audit-regression','exec'),namespace)
    for enabled,local,remote,pending in [(False,0,0,0),(False,240,0,0),(True,240,10,0),(True,240,10,5),(True,255,0,14)]:
        table=torch.zeros((2,300),dtype=torch.int32);slot=torch.zeros_like(table);tier=torch.full_like(table,-1)
        table[1,:local]=torch.arange(1,local+1,dtype=torch.int32)
        slot[1,:local]=table[1,:local];tier[1,:local]=0
        slot[1,local:local+remote]=torch.arange(remote,dtype=torch.int32);tier[1,local:local+remote]=1
        state=NS(enabled=enabled,live={1:local+remote} if local+remote else {},tier=tier,slot=slot,
            remote_counts=Counter({0:remote}) if remote else Counter(),raw=NS(device='cpu',req_to_token_pool=NS(req_to_token=table)),cfg=dict(kv_capacity_tokens=256),
            physical_allocator=NS(available_size=lambda n=local:256-n),committed_remote=remote,
            remote_positions={(0,i):i for i in range(remote)},peer=NS(shape=(36,2,64,8,128)),
            peer_free=list(range(64-remote-pending)),pending=object() if pending else None,pending_targets=list(range(pending)))
        result=namespace['audit'](state)
        assert (result['local_active_tokens'],result['remote_active_tokens'])==(local,remote)
    print('BOOKKEEPING_REGRESSION_PASSED: empty,native,committed-peer,in-flight-peer reservations')


if __name__=='__main__':main()
