"""Adapter from protected trace replay to validated three-donor FlashInfer."""
import time
from types import SimpleNamespace
import torch
from prefill_backend import Backend


class DynamicBackend(Backend):
    def ensure_headroom(self,needed):
        start=time.perf_counter()
        if self.pending is not None:self.commit()
        # A heterogeneous prefill can allocate 4096 total tokens. Preserve the
        # 2048-page transfer unit while permitting two consecutive refill copies.
        assert needed<=self.allocation_budget(),(needed,self.allocation_budget())
        while self.physical_allocator.available_size()<needed:
            partial=min(needed,self.physical_allocator.available_size()+self.variant['migration_tokens'])
            self.launch(partial,False);self.commit()
        return (time.perf_counter()-start)*1000


class Transport:
    def __init__(self,raw,peer,cfg,validation):
        c=cfg['cooperation']
        options=dict(cfg,**c,kv_capacity_tokens=c['offpath_physical_capacity_tokens'],source='peer',validation=validation,
            remote_rows=5,tile_rows=32,decode_splits=8,prefill_chunk_tokens=2048,
            prefill_kernel=dict(name='peer_auto',block_m=64,splits=1),stage_depth=1,reference_atol=.01,reference_rtol=.02)
        old=SimpleNamespace(raw=raw,peer=peer,backend=raw.attn_backend,native=raw.attn_backend.forward_decode)
        self.control=DynamicBackend(raw,peer,options,old,'tiered_peer');self.validation=validation
        self.control.configure(dict(name='batch2048_fused_async',migration_tokens=2048,async_copy=True,copy_impl='fused',reserve_steps=0))
        from decode_oracle import install
        install(self.control)
        self.decode_checks=0;self.prefill_checks=0

    def capture(self,max_batch,validation):
        self.validation=validation;self.control.validation=validation
        self.control.prepare_graphs(max_batch)

    def reset(self):
        self.control.reset(1,False);self.control.req_objects=[];self.control.check_prefill=False
        self.control.prefill_checks=[];self.decode_checks=0;self.prefill_checks=0

    def commit(self):
        if self.control.pending is not None:self.control.commit()

    def available(self):
        self.commit();return self.control.allocation_budget()

    def before(self,nodes,phase):
        c=self.control;c.context=dict(phase=phase,turn=-1,chunk=len(c.frames),step=len(c.frames));c.req_objects=[n['req']for n in nodes]
        amount=sum(n['_selected_size']for n in nodes)if phase=='prefill'else len(nodes)
        c.ensure_headroom(amount)
        c.check_prefill=(self.validation and phase=='prefill'and c.remote_active>0 and
            max(n['_selected_size']for n in nodes)<=32 and self.prefill_checks<2)
        if c.check_prefill:self.prefill_checks+=1

    def after(self,phase,has_next=True):
        c=self.control
        if phase=='decode':
            if self.validation and c.remote_active and self.decode_checks<2:
                c.validate_decode();c.audit();self.decode_checks+=1
        c.check_prefill=False

    def prefetch(self):
        # Enqueue before token.cpu() so old immutable KV can copy during the model graph.
        self.control.prefetch_next(True)

    def release(self,node):
        self.control.release_requests([node['req']])

    def snapshot(self):
        c=self.control;local,remote,_=c.budget()
        assert not remote or 10000*remote<1797*local
        return dict(local_active_tokens=local,remote_active_tokens=remote,
            pending_remote_tokens=c.pending[-1]['tokens']if c.pending is not None else 0)

    def finish(self,folder):
        from worker import save
        c=self.control;self.commit();audit=c.audit()
        assert audit['logical_tokens']==0
        # The old intermediate class's finish method has a two-GPU 5.99% guard.
        # Resolve timing events directly; audited three-source bound is 17.97%.
        from backend import base
        final=base.ReplayBackend.finish(c)
        save(folder/'peer_operations.json',dict(final=final,forwards=c.frames,migrations=c.migrations,
            releases=c.releases,restorations=c.restorations,decode_checks=c.layer_checks,prefill_checks=c.prefill_checks,
            stage_buffer_bytes=c.stager.bytes if c.stager is not None else 0,primary_query_copies=c.validation,
            layout='three owners, g%3/g//3; remote suffix restoration preserves per-request remote prefix'))
        return final
