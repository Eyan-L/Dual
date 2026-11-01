"""Striped global remote slots g map to donor g%3 and physical slot g//3."""
import torch


class Plane:
    def __init__(self,bank,layer,plane):self.bank,self.layer,self.plane=bank,layer,plane

    def index_select(self,dimension,indices):
        assert dimension==0
        out=torch.empty((indices.numel(),8,128),device=indices.device,dtype=torch.bfloat16)
        # Numerical/audit path only. Production attention/copies resolve pointers in a single kernel.
        for d,source in enumerate(self.bank.banks):
            mask=indices%3==d
            out[mask]=source[self.layer,self.plane].index_select(0,indices[mask]//3)
        return out


class RemoteBank:
    def __init__(self,banks):
        assert len(banks)==3 and all(v.shape==banks[0].shape for v in banks)
        self.banks=list(banks);self.shape=(36,2,3*banks[0].shape[2],8,128)
        self.device=banks[0].device;self.dtype=banks[0].dtype
        self.pointers=torch.tensor([bank[layer,plane].data_ptr() for bank in banks for layer in range(36) for plane in range(2)],device=self.device,dtype=torch.int64)

    def __getitem__(self,key):
        layer,plane=key
        return Plane(self,layer,plane)

    def numel(self):return sum(v.numel() for v in self.banks)
    def element_size(self):return self.banks[0].element_size()
    def fill_(self,value):
        for bank in self.banks:bank.fill_(value)
        return self
    def copy_(self,other):
        for dst,src in zip(self.banks,other.banks):dst.copy_(src)
        return self

    def local_shadow(self):return RemoteBank([torch.empty_like(v) for v in self.banks])
