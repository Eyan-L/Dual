"""One-layer ring,CUDA asynchronous contiguous peer copies,explicit fences."""
import torch


class LayerStager:
    def __init__(self,control):
        self.control=control
        self.buffer=torch.empty_like(control.peer[0])
        self.stream=torch.cuda.Stream(device=control.raw.device)
        self.ready=[torch.cuda.Event() for _ in range(36)]
        self.consumed=[torch.cuda.Event() for _ in range(36)]
        self.input_ready=torch.cuda.Event()
        self.copy_events=[]

    @property
    def bytes(self):return self.buffer.numel()*self.buffer.element_size()

    def start(self):
        self.input_ready.record();self.stream.wait_event(self.input_ready)
        self.copy_events=[];self.audit_positions=None
        if self.control.check_prefill:
            self.audit_positions=torch.tensor(sorted(self.control.remote_positions.values()),device=self.control.raw.device,dtype=torch.int64)
        self.launch(0)

    def launch(self,layer):
        c=self.control
        with torch.cuda.stream(self.stream):
            a=torch.cuda.Event(enable_timing=True) if c.profile else None
            if a is not None:a.record()
            # Both planes and all remote slots are contiguous. No gather,
            # address-map rewrite,SM copy kernel or whole36-layer local mirror.
            self.buffer.copy_(c.peer[layer],non_blocking=True)
            self.ready[layer].record()
            if a is not None:
                z=torch.cuda.Event(enable_timing=True);z.record();self.copy_events.append((layer,a,z))

    def wait(self,layer):
        torch.cuda.current_stream().wait_event(self.ready[layer])
        if self.audit_positions is not None:
            self.ready[layer].synchronize()
            for plane in range(2):
                assert torch.equal(self.buffer[plane].index_select(0,self.audit_positions),
                    self.control.peer[layer,plane].index_select(0,self.audit_positions)),('Staged payload differs',layer,plane)
        return self.buffer

    def next(self,layer):
        if layer==35:return
        # The paged attention is the last reader of this ring buffer. Native
        # LSE merge/KV writes/O/MLP remain on main stream and can overlap next copy.
        self.consumed[layer].record();self.stream.wait_event(self.consumed[layer]);self.launch(layer+1)
