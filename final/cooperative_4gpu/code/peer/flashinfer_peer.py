"""FlashInfer0.6.18 FA2 with native planning/math and three-donor tiered paged-KV addresses.

Only an isolated JIT module is changed. Installed FlashInfer and old launches
remain untouched. Encoded pages0..C address local storage;C+1+g addresses donor g%3, physical slot g//3.
"""
import functools
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil

import torch


TENSORS=['peer_k0','peer_v0','peer_k1','peer_v1','peer_k2','peer_v2','unified_indices']
TENSOR_TYPES=['DTypeKV']*6+['int32_t']
SCALARS=['local_capacity','sm_scale']
SCALAR_TYPES=['int64_t','double']


def replace_once(text,old,new):
    assert text.count(old)==1,(old[:100],text.count(old))
    return text.replace(old,new)


def patch_page(text):
    marker='  DType* v_data;'
    addition='''
#ifdef SLACK_FI_TIERED_KV
  DType* peer_k0_data = nullptr;
  DType* peer_v0_data = nullptr;
  DType* peer_k1_data = nullptr;
  DType* peer_v1_data = nullptr;
  DType* peer_k2_data = nullptr;
  DType* peer_v2_data = nullptr;
  uint32_t local_capacity = 0;

  // Absolute byte addresses preserve arbitrary CUDA IPC virtual addresses.
  // Avoid expressing cross-allocation addresses as C++ array-relative offsets.
  __device__ __forceinline__ size_t tiered_address(
      IdType page_iter,uint32_t head_idx,uint32_t entry_idx,uint32_t feat_idx,
      IdType last_indptr,bool value) const {
    const IdType page = page_iter < last_indptr ? __ldg(indices+page_iter) : 0;
    const bool far = static_cast<uint32_t>(page) > local_capacity;
    const uint32_t remote = static_cast<uint32_t>(page)-local_capacity-1;
    const uint32_t bank = remote % 3;
    const size_t physical = far ? remote / 3 : page;
    DType* pk = bank == 0 ? peer_k0_data : (bank == 1 ? peer_k1_data : peer_k2_data);
    DType* pv = bank == 0 ? peer_v0_data : (bank == 1 ? peer_v1_data : peer_v2_data);
    DType* base = value ? (far ? pv : v_data) : (far ? pk : k_data);
    const size_t offset = value ? get_v_elem_offset(physical,head_idx,entry_idx,feat_idx)
                                : get_elem_offset(physical,head_idx,entry_idx,feat_idx);
    return reinterpret_cast<size_t>(base)+offset*sizeof(DType);
  }
#endif
'''
    # Only the first paged_kv_t, not MLA or other cache structs, is extended.
    assert text.count(marker)==1
    text=text.replace(marker,marker+'\n'+addition)
    for method,is_v in [('protective_get_kv_offset',False),('protective_get_v_offset',True)]:
        start=text.index('size_t '+method+'(');brace=text.index('{',start)
        text=text[:brace+1]+f'\n#ifdef SLACK_FI_TIERED_KV\n    return tiered_address(page_iter,head_idx,entry_idx,feat_idx,last_indptr,{str(is_v).lower()});\n#else\n'+text[brace+1:]
        # Existing method has one if/else block; match braces to its end.
        depth=1;end=brace+1
        while depth:
            end+=1
            if text[end]=='{':depth+=1
            elif text[end]=='}':depth-=1
        text=text[:end]+'\n#endif\n  '+text[end:]
    for old,kind in [
        ('return k_data + protective_get_kv_offset(page_iter, head_idx, entry_idx, feat_idx, last_indptr);','protective_get_kv_offset'),
        ('return v_data + protective_get_v_offset(page_iter, head_idx, entry_idx, feat_idx, last_indptr);','protective_get_v_offset')]:
        text=replace_once(text,old,f'#ifdef SLACK_FI_TIERED_KV\n    return reinterpret_cast<DType*>({kind}(page_iter,head_idx,entry_idx,feat_idx,last_indptr));\n#else\n    {old}\n#endif')
    for plane,is_v,offset in [('k',False,'get_elem_offset'),('v',True,'get_v_elem_offset')]:
        old=f'return {plane}_data + {offset}(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);'
        text=replace_once(text,old,f'#ifdef SLACK_FI_TIERED_KV\n    return reinterpret_cast<DType*>(tiered_address(page_iter,head_idx,entry_idx,feat_idx,page_iter+1,{str(is_v).lower()}));\n#else\n    {old}\n#endif')
    return text


def patch_prefill(text):
    helper='''
template <typename DType>
__device__ __forceinline__ DType* slack_paged_address(DType* base,size_t offset) {
#ifdef SLACK_FI_TIERED_KV
  return reinterpret_cast<DType*>(offset);
#else
  return base+offset;
#endif
}
'''
    text=replace_once(text,'namespace flashinfer {','namespace flashinfer {\n'+helper)
    old='DType* gptr = kv_ptr + thr_local_kv_offset[i];';assert text.count(old)==2
    text=text.replace(old,'DType* gptr = slack_paged_address(kv_ptr,thr_local_kv_offset[i]);')
    old='''DType* gptr = kv_ptr + get_paged_kv_offset_for_logical_row<produce_v, KTraits>(
                                 paged_kv, packed_page_iter_base, last_indptr, kv_head_idx,
                                 logical_row, lane_idx);'''
    assert text.count(old)==2
    text=text.replace(old,'''DType* gptr = slack_paged_address(kv_ptr,get_paged_kv_offset_for_logical_row<produce_v, KTraits>(
                                 paged_kv, packed_page_iter_base, last_indptr, kv_head_idx,
                                 logical_row, lane_idx));''')
    return text


@functools.cache
def module():
    import flashinfer.jit.env as env
    from flashinfer.jit.attention import gen_customize_batch_prefill_module
    from flashinfer.prefill import get_batch_prefill_jit_module
    assert importlib.metadata.version('flashinfer-python')=='0.6.18'
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:14]
    uri='slack_peer3_fi0618_'+digest
    build=Path(os.environ['SLACK_FI_BUILD_ROOT'])/uri
    include=build/'include';target=include/'flashinfer'
    if not (build/'PATCH_READY').exists():
        shutil.copytree(env.FLASHINFER_INCLUDE_DIR/'flashinfer',target,dirs_exist_ok=True)
        for relative,patch in [('page.cuh',patch_page),('attention/prefill.cuh',patch_prefill)]:
            p=target/relative;p.write_text(patch(p.read_text()))
        (build/'PATCH_READY').write_text(digest+'\n')
    spec=gen_customize_batch_prefill_module('fa2',uri,torch.bfloat16,torch.bfloat16,torch.bfloat16,
        torch.int32,128,128,TENSORS,TENSOR_TYPES,SCALARS,SCALAR_TYPES,
        'DefaultAttention<false,false,false,false>',
        '#include<flashinfer/attention/variants.cuh>')
    # The native host launch receives extra pointers via the supported custom
    # JIT argument mechanism; native planner and reduction code stay intact.
    host=next(p for p in spec.sources if p.name=='batch_prefill.cu')
    source=host.read_text();marker='params.paged_kv = paged_kv;'
    addition='''
        params.paged_kv.peer_k0_data = static_cast<DTypeKV*>(peer_k0.data_ptr());
        params.paged_kv.peer_v0_data = static_cast<DTypeKV*>(peer_v0.data_ptr());
        params.paged_kv.peer_k1_data = static_cast<DTypeKV*>(peer_k1.data_ptr());
        params.paged_kv.peer_v1_data = static_cast<DTypeKV*>(peer_v1.data_ptr());
        params.paged_kv.peer_k2_data = static_cast<DTypeKV*>(peer_k2.data_ptr());
        params.paged_kv.peer_v2_data = static_cast<DTypeKV*>(peer_v2.data_ptr());
        params.paged_kv.local_capacity = local_capacity;
        params.paged_kv.indices = static_cast<IdType*>(unified_indices.data_ptr());
'''
    host.write_text(replace_once(source,marker,marker+addition))
    spec.extra_include_dirs=[include]+(spec.extra_include_dirs or [])
    spec.extra_cuda_cflags+=['-DSLACK_FI_TIERED_KV=1']
    loaded=spec.build_and_load()
    return get_batch_prefill_jit_module(uri,loaded),dict(uri=uri,build=str(build),tensors=TENSORS,scalars=SCALARS)


def activate(wrapper):
    wrapper._jit_module,metadata=module()
    wrapper._jit_additional_tensor_names=list(TENSORS)
    wrapper._jit_additional_scalar_names=list(SCALARS)
    wrapper.slack_peer_metadata=metadata
    return wrapper
