# Fork optimizations (brosequist/llama-cpp-turboquant)

This fork is [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (llama.cpp with TurboQuant KV-cache
types), plus fixes and performance work for one specific setup: **two RDNA4 GPUs with different architectures in one
process**, serving through llama-swap in a k3s cluster.

| | |
|---|---|
| GPUs | RX 9070 XT (navi48, gfx1201, 16 GB), plus RX 9060 XT (navi44, gfx1200, 16 GB, half the memory bandwidth) |
| Host | Ryzen 9 7900X, Ubuntu with the in-tree amdgpu driver (kernel 7.0) |
| Main model | Qwen3.8-27B IQ4_XS (`qwen35` arch: hybrid GDN + attention, MTP head). 262K context, turbo4 KV, layer split across both GPUs (`--tensor-split 1.3,1`) |
| Second dual-GPU model | North-Mini-Code-1.0 UD-Q4_K_XL (`cohere2moe`). 262K context, q8_0 KV, `--device ROCm1,ROCm0 --tensor-split 1,1` |

## Overall: before vs after

Qwen3.8-27B with the production flags. **Before** is `rocm-1186790` / `rocm-8dc12a0`, deployed 2026-10-24 before
this work, without MTP. **After** is `exp/qwen-q5k` (deployed 2026-10-26 as `app-7f2bd32`: `deploy/rocm101-stack` + the MoE and decode kernel work below) plus the navi48 clock watcher and `--tensor-split 1.3,1`,
measured back to back under the same GPU conditions (see the note on navi48 clocks below).

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Decode, code prompt (t/s) | ~21 (no speculation) | **54.8** | **~2.6×** |
| Decode, prose prompt (t/s) | ~21 | **42.2** | ~2.0× |
| Decode, 9.3K-token summarisation (t/s) | ~21 | **42.1** | ~2.0× |
| Prefill, 4096-token prompt (`llama-bench pp4096`, t/s) | 808 | **~1520** | ~1.9× |
| Prefill, 9.3K-token prompt through llama-server with MTP (t/s) | 754 | **~1450** | ~1.9× |
| Prefill @ 64K context depth (t/s) | 223 | 492 | 2.2× |
| Prefill @ 224K context depth (t/s) | 152 | 383 | 2.5× |
| KL divergence vs f16 KV (turbo4) | 0.0080 | 0.0065 | lower is better |
| VRAM, navi44 / navi48 (GB) | 14.3 / 10.8 (unbalanced) | 13.1 / 13.4 | balanced; ~2.7 GB headroom each |

North-Mini-Code decode with the q8_0 KV cache (long-context rows: `llama-bench tg64`; edit/fresh rows: llama-server, temp 0).

| Workload | Before (t/s) | After (t/s) | Change |
|---|---:|---:|---:|
| Decode @ 32K context | 31.9 | 46.5 | +46% |
| Decode @ 64K context | 23.8 | 43.3 | +82% |
| Rewrite a 200-line file with a small change | 57.1 | 139.9 (157.6 live) | **~2.5×** |
| Write new code from scratch | 58.4 | 58.0 | unchanged |
| Write new code from scratch, single-token MoE routing (deployed 2026-10-26) | 58.2 | **73.3** | **+26%** |
| Write new code from scratch, + fused MoE sum and `--device ROCm1,ROCm0` (deployed 2026-10-26) | 58.2 | **80.9** | **+39%** |
| Rewrite a file, first 1200 tokens, single-token MoE routing (deployed) | 163.7 | **180.8** | +10% |

The last two rows are a separate same-session A/B (1200-token completions, `--spec-type ngram-simple`); see the Decode table.

Before is the previous production build (`rocm-db6212f`, ROCm 7.2.1), after is the deployed build, back to back. Most of
the gain is [#30803](https://github.com/ggml-org/llama.cpp/pull/30803): on ROCm 10.1 alone it measured 41.5 → 52.6 @32K,
36.3 → 48.7 @64K and 25.5 → 42.8 @128K. North was also unable to run dual-GPU at all before the mixed-arch fixes in the
next section.

> **navi48 clocks skew measurements.** navi48 also drives the desktop. While a desktop session was active it held
> about 1.3 GB more VRAM and kept the GPU out of its deep idle states, and the same binaries ran 5–12% faster
> (9.3K prefill 1379 vs 1312 t/s, code decode 51.9 vs 46.6). Comparisons in this document come from runs taken
> back-to-back under one condition; an earlier ROCm 10.1 result (+5.6% prefill, +11–20% decode) did not hold up
> under that rule and is corrected below.

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
| **ROCm 7.2.1 → 10.1.0** base image (TheRock build) + batch-1 cherry-picks, same conditions | 9.3K with MTP | 1307 | 1303–1321 (flat) |

### Decode

| Step | Prompt: summary / code / prose | Before (t/s) | After (t/s) |
|---|---|---:|---:|
| **MTP self-speculation** (`--spec-type draft-mtp`, NextN head from the main GGUF) | code / prose | 21.5 / ~21 | 41.5 / 32.4 |
| **`--tensor-split 1.2,1`** | summary / code / prose | 29.98 / 40.76 / 33.37 | 32.63 / 41.01 / 34.63 |
| ggml-org **[#31037](https://github.com/ggml-org/llama.cpp/pull/31037)** port: trimmed MTP draft head (`LLAMA_MTP_DRAFT_VOCAB=65536`). The draft only proposes ids below 65536 and the target still verifies, so output is unaffected | summary / code / prose | 32.63 / 41.01 / 34.63 | 34.72 / 45.56 / 37.13 |
| **ROCm 7.2.1 → 10.1.0** + batch 1, same conditions | summary / code / prose | 33.1 / 48.1 / 35.1 | 35.0 / 46.6 / 37.1 |
| **[#30803](https://github.com/ggml-org/llama.cpp/pull/30803) port**: RDNA4 block-per-lane q8_0/q4_0 K plus GQA head batching in the vec FA kernel. North only; see the table above | North @64K | 23.8 | 43.3 |
| **navi48 DPM `high` only while serving** (host service, not fork code). The split leaves navi48 idle between bursts and its DPM drops to ~1.6 GHz; a watcher polls both GPUs' busy % every 250 ms, sets `high` on load and `auto` 30 s after. Idle power stays ~15 W instead of 38 W | summary / code / prose; 9.3K prefill | 35.2 / 46.6 / 37.1; 1350 | 39.1 / 51.5 / 41.0; 1394 |
| **`--spec-type ngram-simple` on North-Mini-Code**: draft-free n-gram speculation. North has no MTP head and the only EAGLE3 draft doesn't load in this fork | North: code edit / fresh code | 57.1 / 58.4 | 139.9 / 58.0 |
| **Single-token MUL_MAT_ID through the MoE MMVQ kernel on RDNA4** (ours, `exp/moe-n1`, deployed 2026-10-26). A rocprofv3 profile of North's fresh-code decode showed the expert matmuls taking ~45% of GPU time at ~100 GB/s: the generic `mul_mat_vec_q` runs 8 warps per output row on RDNA4, so North's short expert rows (Q5_K down K = 768, Q4_K gate/up K = 2048) leave most lanes idle. One-token calls now use the fork's dedicated `mul_mat_vec_q_moe` kernel (one warp per 2 rows per expert slot), which already served 2–8 tokens, including its fused SwiGLU gate/up form. Microbenchmark on navi48: Q5_K down 146 → 86 µs, Q4_K gate/up 86 → 65 µs (navi44 flat). `GGML_CUDA_MMVQ_MOE_N1=0` restores the old routing. Only MoE models are affected; Qwen3.8 is dense | North: code edit (first 1200 tokens) / fresh code | 163.7 / 58.2 | 180.8 / 73.3 |
| **Fused decode MoE expert sum** (ours on top of [#25952](https://github.com/ggml-org/llama.cpp/pull/25952), `exp/moe-sum`, deployed 2026-10-26). For one token the graph sums the weighted expert outputs as mul → cont(permute) → sum_rows, and on navi44 sum_rows alone cost 10–12 µs per layer. A new matcher recognises that 4-node form and runs it through #25952's weighted-reduction kernel. `GGML_CUDA_MOE_WEIGHTED_REDUCTION=0` disables both forms. Summation order changes, so output text can differ slightly | North `llama-bench tg128`, empty / 16K context | 74.7 / 68.5 | 79.4 / 71.6 |
| **`--device ROCm1,ROCm0` for North** (launch flag). North ties its output layer to the token embedding (570 MB), and llama.cpp puts it on the last device. With the default order that is navi44, where it costs 1.8 ms per token. Reversing the order moves it to navi48; the split ratio follows the device list. VRAM peak 13.9 GB navi44 / 14.4 GB navi48. Rejected for Qwen3.8: it either fills navi48 or costs 17% prefill | North `llama-bench tg128`, empty / 16K context; llama-server fresh code | 79.4 / 71.6; 73.3 | 83.3 / 74.0; **80.9** |
| **`--tensor-split 1.3,1`** for Qwen3.8 (was 1.2,1), re-tuned after the GDN kernel, `-b 8192`, the clock watcher and the new build. 1.1 / 1.4: 1334 / 1482 prefill, but 1.4 lost short-prompt prefill (254 → 188); `-ub` 384 / 512 were slower than 256. North stays at `1,1`: 1.4,1 gains ~3% on fresh code but leaves navi48 ~0.4 GB free, and navi48 also drives the desktop | summary / code / prose; 9.3K prefill | 38.9 / 51.5 / 41.0; 1371 | 39.3 / 52.1 / 41.5; 1451 |
| **Branchless Q4_K/Q5_K scale unpack in mmvq** (vecdotq part of [#26705](https://github.com/ggml-org/llama.cpp/pull/26705), `exp/qwen-q5k`, deployed 2026-10-26 as `app-7f2bd32`). A rocprofv3 profile of Qwen3.8 decode showed the weight matmuls already at 91–96% of bandwidth except Q5_K at the 4-column MTP verify width: 48% on navi48, 62% on navi44, because the scale unpack branched on `j` and re-ran for every column. The compiler now hoists it | Qwen3.8 + MTP: summary / code / prose | 39.3 / 52.1 / 41.5 | 41.9 / 54.6 / 42.1 |
| **q8_1 quantize cache shared across weight types** (ours, `exp/qwen-q5k`). The cache was keyed on the weight type, which the q8_1 layout doesn't depend on, so Qwen3.8's Q5_K + IQ4_XS projections on one input each re-quantized it (~80 launches per token). Outputs token-identical | Qwen3.8 + MTP: summary / code / prose | 41.9 / 54.6 / 42.1 | 42.1 / 54.8 / 42.3 |
| **Gate/up + GLU fused for 2–4-token batches on RDNA4, and row-broadcast operands in the elementwise chain** (ours, `exp/qwen-q5k`). MTP verify batches ran gate, up and GLU as three kernels, and GDN's ADD → SOFTPLUS → MUL as three. `GGML_CUDA_MMVQ_FUSE_NCOLS=0` disables the first. Outputs token-identical | Qwen3.8 + MTP: summary / code / prose | 42.1 / 54.8 / 42.3 | 42.2 / 54.9 / 42.4 |

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
| `power_dpm_force_performance_level=high` on navi48 permanently | +11% MTP decode, but 38 W idle instead of 8 W around the clock. Replaced by the busy-triggered watcher above |
| Other n-gram modes on North (code edit / fresh) | ngram-mod 109 / 58, ngram-map-k4v 91 / 58, ngram-map-k 80 / 58, ngram-cache 90 / 52 (slower on fresh) vs ngram-simple 140 / 58 |
| ggml-org **[#30804](https://github.com/ggml-org/llama.cpp/pull/30804)** + **[#31053](https://github.com/ggml-org/llama.cpp/pull/31053)** (chunked CUDA graphs), batch 2a | No change at any chunk size (0 / 256 / 512 / 1024): 9.3K prefill 1366–1367 vs 1371, decode within 0.4%. Graph launch overhead is not a bottleneck here. Branch `exp/cp-batch2a` |
| ggml-org **[#29353](https://github.com/ggml-org/llama.cpp/pull/29353)** (chunked tensor-core GDN), batch 2b | Ported with RDNA4 WMMA tiles; correct (GDN 52/52, PPL 6.7491 → 6.7452) but not faster at Qwen3.8's shape (16 K / 48 V heads): navi48 5% slower at 256 tokens and 23–31% slower at 1–4K; navi44 7% faster at 256. GDN is ~10% of prefill. Mainline enables it only on RDNA3.5 at ≥ 2048 tokens. Branch `exp/cp-batch2b` |
| ggml-org **[#30087](https://github.com/ggml-org/llama.cpp/pull/30087)** (4 GDN columns per warp), batch 2b | Not needed: it changes the legacy column kernel, and this fork defaults to the row-per-warp port, which already processes 4 columns per warp |
| ggml-org **[#25952](https://github.com/ggml-org/llama.cpp/pull/25952)** (fused MoE weighted expert reduction), batch 3 | North fresh decode 58.2 → 58.6 (+0.6%, noise): on its own it only matches the multi-token form. Changes summation order, so outputs diverge. Its kernel is reused by the deployed fused decode MoE sum above. Branch `exp/cp-batch3` |
| Batch 3 PRs that don't apply to North | [#29184](https://github.com/ggml-org/llama.cpp/pull/29184) (shared experts) and [#29181](https://github.com/ggml-org/llama.cpp/pull/29181) (grouped top-k): North has neither. [#27978](https://github.com/ggml-org/llama.cpp/pull/27978): only adds n_expert_used = 10; North's 8 already has the fast path. [#30517](https://github.com/ggml-org/llama.cpp/pull/30517): low-bit MMVQ, needs earlier mainline MMVQ refactors and only touches Q6_K among North's types. [#27621](https://github.com/ggml-org/llama.cpp/pull/27621) (multi-token MoE fusion) is deferred: mainline refactored the MoE MMVQ kernel first, and multi-token batches already use the faster kernel |
| RDNA4 small-K MMVQ (rows per block = warps per block for short rows) | Q4_K gate/up 129 → 87 µs on navi48 alone, but no end-to-end gain on top of the single-token MoE routing (73.3 vs 73.3 t/s) |
| `HIP_FORCE_DEV_KERNARG=1` (Qwen3.8 decode) | No change (52.14 vs 52.16 code) |
| `--backend-sampling` (Qwen3.8) | Doesn't engage: this ROCm build lacks TOP_K / ARGSORT for a 248K-token vocabulary, so sampling falls back to the CPU; −5% prefill on a 9.3K prompt |
| EAGLE3 draft for North (community GGUF) | Doesn't load: TheTom's EAGLE3 loader predates mainline's (`extract_layers` key, full d2t matrix). Real-world acceptance of that draft is ~1.28 anyway |

## Environment knobs added by this fork

Every behaviour change below can be turned off at run time, which is how each one was A/B'd.

| Variable | Default | Effect |
|---|---|---|
| `GGML_CUDA_MMVQ_MOE_N1=0` | on | Single-token `MUL_MAT_ID` goes back to the generic mmvq kernel (RDNA4 only) |
| `GGML_CUDA_MOE_WEIGHTED_REDUCTION=0` | on | Disables the fused MoE weighted expert sum (#25952 and the decode-form matcher) |
| `GGML_CUDA_MMVQ_FUSE_NCOLS=0` | on | Gate/up + GLU fusion stays single-token only (RDNA4 only) |
| `GGML_CUDA_NO_MIXED_ARCH_WORKAROUND=1` | off | Skips the chunked mmf/mmvf routing for mixed-arch GEMMs and uses rocBLAS (needs ROCm ≥ 7.12; slower here) |
| `GGML_CUDA_GDN_LEGACY=1` | off | Uses the original column-per-thread GATED_DELTA_NET kernel instead of the row-per-warp port |
| `LLAMA_MTP_DRAFT_VOCAB=<rows>` | off | Keeps only the first rows of the MTP draft's output head (#31037 port); production uses 65536 |
| `LLAMA_MTP_PROF=1` | off | Logs per-batch target decode vs MTP `process()` / draft time (adds syncs) |

TheTom's own knobs (`GGML_CUDA_Q8CACHE`, `GGML_CUDA_FUSE_CHAIN`, `TURBO_*`, …) are listed in AGENTS.md.

## Branches

**`bump/thetom-1021-mainline` (the default branch) is the production code**: after each deploy, the deployed branch is
merged into it (first done 2026-10-26 for `exp/qwen-q5k` / `app-7f2bd32`). Before that it carried only these docs on
top of the 2026-10-24 code.


`deploy/rocm101-stack` = `exp/gdn-rows-per-warp` → `exp/mtp-no-batch-sync` → `exp/rocm-10.1` → `exp/cp-batch1` →
[#30803](https://github.com/ggml-org/llama.cpp/pull/30803) (`exp/fattn-vec-rdna4-30803`). Earlier steps are on `exp/turbo-innerq-per-device`, `exp/mmvf-single-launch`,
`deploy/qwen38-mtp-tune`, `fix/mmq-oob-29953`, `fix/multigpu-mixed-arch-mmf-gemm` and
`fix/cohere2moe-mixedarch-router-mmvf`.

`exp/moe-n1` = `deploy/rocm101-stack` + the single-token MoE routing. `exp/moe-sum` = `exp/moe-n1` + #25952 + the decode-form matcher (deployed 2026-10-26). `exp/qwen-q5k` = `exp/moe-sum` + the branchless Q4_K/Q5_K unpack, the cross-type q8 cache and the two multi-token fusions (deployed 2026-10-26 as `app-7f2bd32`). Rejected experiments: `exp/cp-batch2a`, `exp/cp-batch2b`, `exp/cp-batch3`, `exp/mmq-glu-fusion-28702`.
