# KVarN-MLA build status (local mla-experiment branch — DO NOT push)

Goal: extend KVarN to quantize the DeepSeek MLA compressed latent (kv_lora_rank
dims) + keep the decoupled RoPE part fp16. Dev model: DeepSeek-V2-Lite
(kv_lora_rank=512, qk_rope_head_dim=64, 27 layers, 64-expert MoE).

## Run prerequisites on this box (sm_120 / Blackwell)
- `VLLM_ATTENTION_BACKEND=TRITON_MLA`  (FlashInfer/FlashMLA fail arch check)
- `VLLM_DISABLE_FLASHINFER_ROPE=1`     (env-gate added to deepseek_scaling_rope.py;
  FlashInfer rope JIT fails on sm_120 -> use native PyTorch rope)
- `kernel_config={'moe_backend':'triton'}`  (MoE kernels)
- `VLLM_USE_FLASHINFER_SAMPLER=0`, `trust_remote_code=True`, `HF_HUB_OFFLINE=1`

## Validated (real, tested)
- ACCURACY (probe): round-trip rotate->sinkhorn->RTN->dequant->unrotate on the
  512 latent in MLACommonImpl.forward_impl (env KVARN_MLA_BITS). bits=4 AND
  bits=2 keep V2-Lite coherent+correct. -> method works on MLA latents.
  Files: kvarn/mla_probe.py + forward_impl hook (mla_attention.py ~line 618).
- CKPT1 savings: packed 512-latent k4 = 402 B/token vs 1152 fp16 = **2.87x**
  (k2 ~4.3x). RoPE-fp16 (128B) is the floor. round-trip cos 0.9997.
- CKPT2 Triton dequant kernel (D=512, V-style, channel-major pack): cos 1.0 vs
  PyTorch ref, max_abs 1.4e-6, runs on sm_120. (/tmp/triton_kvarn_mla_test.py)
  Note: KVarN's existing `_kvarn_dequant_blocks_kernel` is D-parameterized and
  also usable at D=512.

## Design (key decisions)
- Latent stored ROTATED (c_KV @ H, H = orthonormal Hadamard-512). Rotation
  invariance: scores q_a.c = (H q_a).(H c); output = H^T (sum w_i (H c_i)).
  => FUSED path: rotate absorbed-query by H once/step, kernel uses rotated
  latent directly (per-token cost = dequant only, no per-token matmul),
  un-rotate the output once/step (or fold H^T into W_UV). MATERIALIZE path:
  just dequant+unrotate to fp16, feed stock attention (no query/output change).
- MLA quirk: V == K == the latent. The grouped MLA kernel loads K once and
  (IS_MLA) reuses it as V -> only ONE dequant load to add.
- Injection slot: triton_decode_attention.py `_fwd_kernel_stage1_grouped`,
  the K load ~line 364 (next to the existing fp8 dequant path ~line 371-372,
  which is the precedent: load quantized + multiply by scale).
- RoPE (qk_rope_head_dim=64) kept fp16, loaded as today (kpe ~line 376).

## Remaining build
### 3a MATERIALIZE (working + measurable first)
1. cache_dtype `kvarn_mla_k4_g128` (+k2); register cache.py + torch_utils.
2. Packed MLA spec: head_size -> packed bytes (256 latent + scales + 128 rope).
   page_size from packed bytes (mirror TQ/MLA spec; see attention.py MLA spec).
3. KVarNMLABackend(MLACommonBackend)/Impl(MLACommonImpl): override
   get_kv_cache_shape (packed), do_kv_cache_update (rotate+sinkhorn+RTN+pack
   store), forward_impl decode (gather attended blocks -> ckpt2 dequant kernel
   -> unrotate -> fp16 scratch in MLA layout -> stock TritonMLA attention).
   Hard part: paged gather + block-table remap into the fp16 scratch.
4. Backend selection: route kvarn_mla_* + use_mla -> KVarNMLABackend.
   (MLA selection is NOT the cuda.py list; it's the use_mla path.)
5. Validate vs fp16 baseline on V2-Lite; measure num_gpu_blocks (savings) + tok/s.

### 3b FULLY FUSED (peak speed, after 3a works)
- Modify `_fwd_kernel_stage1_grouped`: replace fp16 K-load with packed-load +
  in-loop dequant (ckpt2 logic) using rotated latent; add query rotation by H
  before the kernel + output un-rotation after (or fold into W_UV at
  process_weights_after_loading). No fp16 materialization.
- Validate numerics vs 3a; measure peak tok/s.

## Honest status
Method + savings + dequant kernel = proven. 3a/3b = substantial multi-session
vLLM-MLA integration + kernel surgery (paged cache, block tables, spec,
backend, selection, then fused attention). Resume from here.
