"""Three separately fenced one-layer buffers;one copy stream per donor."""
from contextlib import nullcontext
import torch


class LayerStager:
    def __init__(self,control):
        self.control=control
        self.buffer=[torch.empty_like(bank[0]) for bank in control.peer.banks]
        self.streams=[torch.cuda.Stream(device=control.raw.device) for _ in range(3)]
        self.ready=[[torch.cuda.Event() for _ in range(36)] for _ in range(3)]
        self.consumed=[torch.cuda.Event() for _ in range(36)]
        self.input_ready=torch.cuda.Event();self.copy_events=[]

    @property
    def bytes(self):return sum(v.numel()*v.element_size() for v in self.buffer)

    def start(self):
        self.input_ready.record();self.copy_events=[];self.audit_positions=None
        if self.control.check_prefill:
            positions=list(self.control.remote_positions.values())
            self.audit_positions=[torch.tensor(sorted(g//3 for g in positions if g%3==d),device=self.control.raw.device,dtype=torch.int64) for d in range(3)]
        for stream in self.streams:stream.wait_event(self.input_ready)
        self.launch(0)

    def launch(self,layer):
        c=self.control
        for d,stream in enumerate(self.streams):
            scope=torch.profiler.record_function(f'SLACK_SUPPLY_L{layer}_D{d}') if c.profile else nullcontext()
            with scope,torch.cuda.stream(stream):
                a=torch.cuda.Event(enable_timing=True) if c.profile else None
                if a is not None:a.record()
                self.buffer[d].copy_(c.peer.banks[d][layer],non_blocking=True)
                self.ready[d][layer].record()
                if a is not None:
                    z=torch.cuda.Event(enable_timing=True);z.record();self.copy_events.append((d,layer,a,z))

    def wait(self,layer):
        for d in range(3):torch.cuda.current_stream().wait_event(self.ready[d][layer])
        if self.audit_positions is not None:
            for d in range(3):
                self.ready[d][layer].synchronize();positions=self.audit_positions[d]
                for plane in range(2):
                    assert torch.equal(self.buffer[d][plane].index_select(0,positions),self.control.peer.banks[d][layer,plane].index_select(0,positions)),('staged payload',d,layer,plane)
        return self.buffer

    def next(self,layer):
        if layer==35:return
        self.consumed[layer].record()
        for stream in self.streams:stream.wait_event(self.consumed[layer])
        self.launch(layer+1)
