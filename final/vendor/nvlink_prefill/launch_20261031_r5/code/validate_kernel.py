"""Small GPU checks: ragged causal GQA, arbitrary tier maps, peer sensitivity."""
import torch
from types import SimpleNamespace
from paged_prefill import prefill
from paged_decode import PagedDecode


def validate(peer):
    device=peer.device;results=[]
    cases=[('local_tail',[0,0,0],[1,37,63],'local'),
           ('mixed_ragged',[65,7],[33,5],'alternating'),
           ('remote_only',[31],[17],'remote'),
           ('long_prefix',[4096],[79],'sparse'),
           ('empty_decode_splits',[0,2],[1,1],'local')]
    for name,prefixes,qlens,placement in cases:
        lengths=[p+q for p,q in zip(prefixes,qlens)];total=sum(lengths);batch=len(lengths)
        local_k=torch.randn((total+1,8,128),device=device,dtype=torch.bfloat16)
        local_v=torch.randn_like(local_k)
        peer_k,peer_v=peer[0,0],peer[0,1]
        peer_k.normal_();peer_v.normal_()
        tier=torch.zeros((batch+1,max(lengths)),device=device,dtype=torch.int32);slot=torch.zeros_like(tier)
        requests=torch.arange(batch-1,-1,-1,device=device,dtype=torch.int32)
        local_next,remote_next=1,0;logical=[]
        for b,length in enumerate(lengths):
            remote=(torch.arange(length,device=device)%2==0 if placement=='alternating' else
                    torch.arange(length,device=device)%20==0 if placement=='sparse' else
                    torch.full((length,),placement=='remote',device=device))
            nr=int(remote.sum());nl=length-nr
            assert remote_next+nr<=peer_k.shape[0]
            row=slot[requests[b],:length];row[remote]=torch.arange(remote_next,remote_next+nr,device=device,dtype=torch.int32)
            row[~remote]=torch.arange(local_next,local_next+nl,device=device,dtype=torch.int32)
            tier[requests[b],:length]=remote.to(torch.int32)
            k=torch.empty((length,8,128),device=device,dtype=torch.bfloat16);v=torch.empty_like(k)
            for dest,local,far in ((k,local_k,peer_k),(v,local_v,peer_v)):
                dest[remote]=far.index_select(0,row[remote].long());dest[~remote]=local.index_select(0,row[~remote].long())
            logical.append((k,v));local_next+=nl;remote_next+=nr
        q=torch.randn((sum(qlens),32,128),device=device,dtype=torch.bfloat16)
        starts=[sum(qlens[:i]) for i in range(batch)]
        args=[torch.tensor(x,device=device,dtype=torch.int32) for x in (lengths,starts,qlens,prefixes)]
        def run():return prefill(q,local_k,local_v,peer_k,peer_v,tier,slot,requests,*args,max(qlens),128**-.5)[0]
        output=run();reference=[]
        for b,(k,v) in enumerate(logical):
            query=q[starts[b]:starts[b]+qlens[b]].float().transpose(0,1)
            key=k.float().repeat_interleave(4,1).transpose(0,1);value=v.float().repeat_interleave(4,1).transpose(0,1)
            score=(query@key.transpose(1,2))*(128**-.5)
            mask=torch.arange(lengths[b],device=device)[None,:]<=prefixes[b]+torch.arange(qlens[b],device=device)[:,None]
            reference.append((score.masked_fill(~mask[None,:,:],float('-inf')).softmax(-1)@value).transpose(0,1))
        expected=torch.cat(reference);assert torch.isfinite(output).all().item()
        torch.testing.assert_close(output.float(),expected,atol=.01,rtol=.02)
        row=dict(name=name,prefixes=prefixes,query_lengths=qlens,placement=placement,
                 max_abs_error=float((output.float()-expected).abs().max()),
                 relative_l2=float((output.float()-expected).norm()/expected.norm()),remote_slots=remote_next)
        raw=SimpleNamespace(device=device,token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda layer:(local_k,local_v)))
        decoder=PagedDecode(raw,peer,tier,slot,batch,splits=4)
        decoder.requests.copy_(requests);decoder.seq.copy_(args[0])
        lastq=torch.stack([q[starts[b]+qlens[b]-1] for b in range(batch)])
        lastref=torch.stack([expected[starts[b]+qlens[b]-1] for b in range(batch)])
        decoded=decoder.run(lastq,0);assert torch.isfinite(decoded).all().item()
        torch.testing.assert_close(decoded.float(),lastref,atol=.01,rtol=.02)
        row['decode_max_abs_error']=float((decoded.float()-lastref).abs().max())
        if placement=='remote':
            peer_v[:remote_next].add_(.5);changed=run()
            torch.testing.assert_close(changed.float()-output.float(),torch.full_like(output.float(),.5),atol=.012,rtol=.02)
            peer_v[:remote_next].sub_(.5)
            # BF16 add/sub can round, so restore the exact original source.
            peer_v[:remote_next].copy_(logical[0][1]);restored=run()
            torch.testing.assert_close(restored,output,atol=0,rtol=0)
            row.update(peer_value_perturbation_changes_output=True,exact_restore=True)
        results.append(row)
    return dict(status='PAGED_PREFILL_KERNEL_CHECKS_PASSED',cases=results,
                reference='FP32 explicit causal softmax(QK^T/sqrt(128))V with GQA expansion; actual CUDA-IPC KV addresses used.')
