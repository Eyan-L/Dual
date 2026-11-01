"""Process-owned CUDA allocations shared through explicit CUDA IPC handles."""
import base64
import ctypes as C
from pathlib import Path


class IpcHandle(C.Structure):
    _fields_ = [('reserved', C.c_ubyte*64)]


class PointerAttributes(C.Structure):
    _fields_ = [('type', C.c_int), ('device', C.c_int),
                ('devicePointer', C.c_void_p), ('hostPointer', C.c_void_p),
                # CUDA 13 driver_types.h includes this ABI tail, zero initialized.
                ('reserved', C.c_long*8)]


class Cuda:
    def __init__(self):
        candidates = ['libcudart.so.13','libcudart.so']
        import torch
        candidates += [str(p) for p in Path(torch.__file__).resolve().parents[1].glob('nvidia/**/libcudart.so*')]
        for candidate in candidates:
            try:
                self.lib = C.CDLL(candidate)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('Cannot locate the container CUDA runtime')
        signatures = {
            'cudaSetDevice': [C.c_int],
            'cudaDeviceCanAccessPeer': [C.POINTER(C.c_int),C.c_int,C.c_int],
            'cudaDeviceEnablePeerAccess': [C.c_int,C.c_uint],
            'cudaMalloc': [C.POINTER(C.c_void_p),C.c_size_t],
            'cudaFree': [C.c_void_p],
            'cudaIpcGetMemHandle': [C.POINTER(IpcHandle),C.c_void_p],
            'cudaIpcOpenMemHandle': [C.POINTER(C.c_void_p),IpcHandle,C.c_uint],
            'cudaIpcCloseMemHandle': [C.c_void_p],
            'cudaPointerGetAttributes': [C.POINTER(PointerAttributes),C.c_void_p],
            'cudaGetLastError': [],
        }
        for name, args in signatures.items():
            function=getattr(self.lib,name);function.argtypes=args;function.restype=C.c_int
        self.lib.cudaGetErrorString.argtypes=[C.c_int]
        self.lib.cudaGetErrorString.restype=C.c_char_p

    def check(self, code):
        if code:
            raise RuntimeError(f'CUDA {code}: '+self.lib.cudaGetErrorString(code).decode())

    def set_device(self, device):
        self.check(self.lib.cudaSetDevice(device))

    def enable_peer(self, source, owner):
        self.set_device(source)
        capable=C.c_int()
        self.check(self.lib.cudaDeviceCanAccessPeer(C.byref(capable),source,owner))
        assert capable.value == 1, 'Allocated GPUs do not support peer access'
        code=self.lib.cudaDeviceEnablePeerAccess(owner,0)
        if code and 'already enabled' in self.lib.cudaGetErrorString(code).decode().lower():
            self.lib.cudaGetLastError()
        else:
            self.check(code)

    def allocate(self, size):
        ptr=C.c_void_p();self.check(self.lib.cudaMalloc(C.byref(ptr),size))
        handle=IpcHandle();self.check(self.lib.cudaIpcGetMemHandle(C.byref(handle),ptr))
        return ptr.value,base64.b64encode(bytes(handle.reserved)).decode()

    def open(self, encoded):
        data=base64.b64decode(encoded);assert len(data)==64
        handle=IpcHandle();C.memmove(C.addressof(handle),data,64)
        ptr=C.c_void_p()
        self.check(self.lib.cudaIpcOpenMemHandle(C.byref(ptr),handle,1))
        return ptr.value

    def attributes(self, address):
        attrs=PointerAttributes()
        self.check(self.lib.cudaPointerGetAttributes(C.byref(attrs),C.c_void_p(address)))
        return dict(memory_type=attrs.type,owner_logical_device=attrs.device,
                    device_pointer=attrs.devicePointer,host_pointer=attrs.hostPointer)

    def close(self, address):
        self.check(self.lib.cudaIpcCloseMemHandle(C.c_void_p(address)))

    def free(self, address):
        self.check(self.lib.cudaFree(C.c_void_p(address)))
