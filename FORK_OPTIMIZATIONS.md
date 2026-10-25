# Fork optimizations (brosequist/llama-cpp-turboquant)

This fork is [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (llama.cpp with TurboQuant KV-cache
types), plus fixes and performance work for one specific setup: **two RDNA4 GPUs with different architectures in one
process**, serving through llama-swap in a k3s cluster.

| | |
|---|---|
| GPUs | RX 9070 XT (navi48, gfx1201, 16 GB), plus RX 9060 XT (navi44, gfx1200, 16 GB, half the memory bandwidth) |
| Host | Ryzen 9 7900X, Ubuntu with the in-tree amdgpu driver (kernel 7.0) |
| Main model | Qwen3.8-27B IQ4_XS (`qwen35` arch: hybrid GDN + attention, MTP head). 262K context, turbo4 KV, layer split across both GPUs (`--tensor-split 1.2,1`) |
| Second dual-GPU model | North-Mini-Code-1.0 UD-Q4_K_XL (`cohere2moe`). 262K context, q8_0 KV, `--tensor-split 1,1` |

## Overall: before vs after

Qwen3.8-27B with the production flags. **Before** is `rocm-1186790` / `rocm-8dc12a0`, deployed 2026-10-24 before
this work, without MTP. **After** is the 2026-10-25 release candidate (`deploy/rocm101-stack`), measured in a test pod; production numbers
will replace these once it is deployed.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Decode, code prompt (t/s) | ~21 (no speculation) | **51.9** | **~2.5×** |
| Decode, prose prompt (t/s) | ~21 | **41.3** | ~2.0× |
| Decode, 9.3K-token summarisation (t/s) | ~21 | **38.9** | ~1.85× |
| Prefill, 4096-token prompt (`llama-bench pp4096`, t/s) | 808 | **1611** | ~2.0× |
| Prefill, 9.3K-token prompt through llama-server with MTP (t/s) | 754 | **1379** | ~1.8× |
| Prefill @ 64K context depth (t/s) | 223 | 492 | 2.2× |
| Prefill @ 224K context depth (t/s) | 152 | 383 | 2.5× |
| KL divergence vs f16 KV (turbo4) | 0.0080 | 0.0065 | lower is better |
| VRAM, navi44 / navi48 (GB) | 14.3 / 10.8 (unbalanced) | 13.5 / 14.2 | balanced; ~2 GB headroom each |

North-Mini-Code: decode at long context with the q8_0 KV cache, `llama-bench tg64`.

| Context depth | Before (t/s) | After (t/s) | Change |
|---|---:|---:|---:|
| 32K | 41.5 | 52.6 | +27% |
| 64K | 36.3 | 48.7 | +34% |
| 128K | 25.5 | 42.8 | +68% |

The north "before" column is already on ROCm 10.1, so the table isolates [#30803](https://github.com/ggml-org/llama.cpp/pull/30803). North was also unable to run
dual-GPU at all before the mixed-arch fixes in the next section.

> **Methodology.** Decode numbers are temperature 0, 700-token completions through llama-server's
> `/v1/chat/completions`, using three prompts: a 9.3K-token wikitext summary, a Python coding task and a 300-word
> prose answer. Prefill numbers are `llama-bench -p 4096 -n 0 -r 3`, or the llama-server `timings` for the 9.3K
> prompt. Where a cell says "~", the baseline came from an earlier harness on the same hardware. Each step below was
> measured A/B in a single session.

## Optimizations, in the order they landed

### Correctness: making the mixed-architecture dual-GPU split work

| Change | Problem | Result |
|---|---|---|
| **Chunked mmf for float GEMMs on mixed archs** (ours, `ggml-cuda.cu`) | rocBLAS/Tensile's code-object cache isn't keyed by GPU arch. With gfx1200 and gfx1201 in one process, a gfx1200 kernel got launched on the gfx1201 → `no kernel image` abort on large prefill | 14B and larger models run split across both GPUs. Single-GPU and same-arch setups are untouched |
| **Chunked mmvf for the F32 MoE router** (ours) | The same bug on cohere2moe's F32 `ffn_gate_inp`, which has no RDNA4 MMA, so mmf declines it | North-Mini-Code runs dual-GPU at 262K. Before, it returned 502s on large prompts |
| ggml-org **[#29953](https://github.com/ggml-org/llama.cpp/pull/29953)** port: MMQ src1 padding | Out-of-bounds reads for 1–7-column batches | Correctness only; no performance change |

ROCm 10.1 fixes the underlying rocBLAS bug (ROCm/rocm-libraries#4781). We keep our workaround anyway because it is
faster (pp4096 1606 vs 1544 t/s without it). `GGML_CUDA_NO_MIXED_ARCH_WORKAROUND=1` turns it off.

### Prefill and long context

| Step | Measured on | Before | After |
|---|---|---:|---:|
| **TheTom bump + 7 mainline cherry-picks**, chiefly **[#28102](https://github.com/ggml-org/llama.cpp/pull/28102)** (MMA flash attention for head dim 256 on RDNA4; qwen3.8 is D=256) | prefill @64K / @224K | 223 / 152 | 492 / 383 |
| **Per-device TurboQuant InnerQ scale** (ours). The shared scale tensor pulled all of navi44's attention WHT ops onto navi48: 39 graph splits instead of 3, and no pipeline overlap | pp (no MTP) / 9.3K with MTP | 808 / 754 | 1240 / 963 |
| **`--tensor-split 1.2,1`** (was 0.88,1). navi48 has twice the bandwidth | 9.3K prefill | 964 | 995 |
| **Single-launch mixed-arch mmvf** (ours). The F32 GDN alpha/beta GEMM went from ~98K eight-column launches per 4K prompt to one launch | pp4096 | 1372 | 1450 |
| **Row-per-warp GATED_DELTA_NET kernel**, a port of [#22587](https://github.com/ggml-org/llama.cpp/pull/22587) onto TheTom's keep_rs/checkpoint contract (ours). It is also closer to a CPU-reference GDN: KLD 0.0046 vs 0.0071 | pp4096 / 9.3K with MTP | 1450 / 1064 | 1491 / 1083 |
| **`-b 8192`** (was 1024). MTP reads `h_nextn` back after every batch, which drains the layer-split pipeline; navi48 then idles and down-clocks to ~1.6 GHz | 9.3K with MTP | 1084 | 1306 |
| **MTP: keep only the last `h_nextn` row for prompt batches** (ours). Every row was being copied into a buffer that is only read for verify batches | 9.3K with MTP | 1291 | 1309 |
| **ROCm 7.2.1 → 10.1.0** base image (TheRock build) | pp4096 / 9.3K with MTP | 1518 / 1309 | 1606 / 1382 |

### Decode

| Step | Prompt: summary / code / prose | Before (t/s) | After (t/s) |
|---|---|---:|---:|
| **MTP self-speculation** (`--spec-type draft-mtp`, NextN head from the main GGUF) | code / prose | 21.5 / ~21 | 41.5 / 32.4 |
| **`--tensor-split 1.2,1`** | summary / code / prose | 29.98 / 40.76 / 33.37 | 32.63 / 41.01 / 34.63 |
| ggml-org **[#31037](https://github.com/ggml-org/llama.cpp/pull/31037)** port: trimmed MTP draft head (`LLAMA_MTP_DRAFT_VOCAB=65536`). The draft only proposes ids below 65536 and the target still verifies, so output is unaffected | summary / code / prose | 32.63 / 41.01 / 34.63 | 34.72 / 45.56 / 37.13 |
| **ROCm 7.2.1 → 10.1.0** | summary / code / prose | 32.5 / 47.1 / 34.5 | 38.0 / 52.4 / 41.4 |
| **[#30803](https://github.com/ggml-org/llama.cpp/pull/30803) port**: RDNA4 block-per-lane q8_0/q4_0 K plus GQA head batching in the vec FA kernel. North only; see the table above | North @128K | 25.5 | 42.8 |

### Mainline cherry-picks (batch 1: correctness and security)

[#28198](https://github.com/ggml-org/llama.cpp/pull/28198) (concurrent streams per split on multi-GPU), [#28475](https://github.com/ggml-org/llama.cpp/pull/28475) (races in mmid/mmf), [#28068](https://github.com/ggml-org/llama.cpp/pull/28068) (GDN q/k l2-norm matches FLA:
`rsqrt(sum+eps)`), [#28149](https://github.com/ggml-org/llama.cpp/pull/28149) (allocation-failure checks), [#29384](https://github.com/ggml-org/llama.cpp/pull/29384) (gguf integer overflow), [#29856](https://github.com/ggml-org/llama.cpp/pull/29856) (recurrent-state reserve
covers every split), [#30135](https://github.com/ggml-org/llama.cpp/pull/30135) (`ggml_acc` OOB write), [#26771](https://github.com/ggml-org/llama.cpp/pull/26771) (stale CUDA graphs after pool flush), [#30836](https://github.com/ggml-org/llama.cpp/pull/30836) (mmf warp
size), [#30969](https://github.com/ggml-org/llama.cpp/pull/30969) (`ggml_time_us` overflow), [#30885](https://github.com/ggml-org/llama.cpp/pull/30885) (gfx12 bf16 WMMA operand type), [#29755](https://github.com/ggml-org/llama.cpp/pull/29755) (grammar recursion limit).
[#29393](https://github.com/ggml-org/llama.cpp/pull/29393) (RMS_NORM+SCALE fusion) is included so that [#28068](https://github.com/ggml-org/llama.cpp/pull/28068)'s extra scale op doesn't add a kernel launch. The batch is
performance-neutral: pp4096 1606 → 1611, and decode is within run-to-run noise.

## Tried and rejected

| Experiment | Result |
|---|---|
| ggml-org **[#28702](https://github.com/ggml-org/llama.cpp/pull/28702)** (fused gate/up + GLU MMQ) | Slower on RDNA4 at every batch size: pp4096 1518 → 1471–1489 at `-ub 256`, −1% at 512 and 1024. Branch `exp/mmq-glu-fusion-28702` |
| `-ub` 128 / 512 / 1024 (256 is production) | pp4096 859 / 1442 / 1327 vs 1496 |
| `--spec-chain 3` | Decode on the summary prompt 33.1 → 29.3 |
| `--spec-draft-n-max` 4 or 5 | Worse than the default of 3 |
| `draft-mtp-adaptive` | 38.0 / 30.4 vs 41.5 / 32.4 |
| Quantised draft KV (`-ctkd/-ctvd q8_0`) | No speedup; +0.5 GB (the draft KV already inherits turbo4) |
| Separate, larger n_ubatch for the MTP draft context | No gain, because the MTP head decode is compute-bound; +0.7–1.9 GB |
| Disabling HIP graphs | About −1% |
| `power_dpm_force_performance_level=high` on navi48 | +11% MTP decode, but 38 W idle instead of 8 W around the clock. Not adopted |

## Branches

`deploy/rocm101-stack` = `exp/gdn-rows-per-warp` → `exp/mtp-no-batch-sync` → `exp/rocm-10.1` → `exp/cp-batch1` →
[#30803](https://github.com/ggml-org/llama.cpp/pull/30803) (`exp/fattn-vec-rdna4-30803`). Earlier steps are on `exp/turbo-innerq-per-device`, `exp/mmvf-single-launch`,
`deploy/qwen38-mtp-tune`, `fix/mmq-oob-29953`, `fix/multigpu-mixed-arch-mmf-gemm` and
`fix/cohere2moe-mixedarch-router-mmvf`.
