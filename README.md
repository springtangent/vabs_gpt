# vabs_gpt

A GPT-style transformer, trained and sampled entirely through the
[vabs](../vabs) Vulkan wrapper. Training runs on byte-level
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Every
forward, backward and optimizer operation is a vabs `@compute_shader`
written in vabs' Python shader DSL. There is no PyTorch, NumPy or CUDA in the
program, only the standard library and vabs.

The model shape and the training run are configurable. With the defaults,
`train` produces a **249,472-parameter** model:

| | Default |
| --- | --- |
| Vocabulary | 256 byte values (byte 0 = end of story) |
| Context | 256 bytes |
| Width / heads / layers / MLP | 64 / 4 / 4 / 256 (GELU), pre-LayerNorm, learned positions, untied head |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0, no clipping |
| Schedule | 10,000 steps × batch 32 × 256 = 8,192 bytes/step, lr 3e-3, warmup 200, cosine to lr/10 |

## Run

From the workspace root (`C:\Users\bafre\universim`):

```powershell
# 1. Once: convert the cached Hugging Face parquet shards to raw bytes (needs pyarrow)
.ptvenv\Scripts\python.exe vabs_gpt\prepare.py          # -> vabs_gpt\data\train.bin, val.bin

# 2. Train (writes vabs_gpt\model.bin after every evaluation)
.venv\Scripts\python.exe -m vabs_gpt train

# 3. Generate
.venv\Scripts\python.exe -m vabs_gpt sample --prompt "Once upon a time" --num 3
```

`.venv` has vabs installed. `train --help`, `sample --help` and `info --help`
list every option with its default.

## Configuring a run

```powershell
# A larger model: 1.29M parameters. --dry-run prints sizes, tokens and device memory, then exits.
.venv\Scripts\python.exe -m vabs_gpt train --dim 128 --layers 6 --heads 8 --dry-run

# Train it: smaller micro-batches with accumulation (16 x 4 = 64 sequences per step),
# a lower peak learning rate, AdamW decay and gradient clipping.
.venv\Scripts\python.exe -m vabs_gpt train --dim 128 --layers 6 --heads 8 `
    --batch 16 --grad-accum 4 --lr 1e-3 --weight-decay 0.1 --grad-clip 1.0 `
    --steps 20000 --out vabs_gpt\big.bin --sample-every 2000

# Continue a run (Ctrl+C saves a checkpoint, so this also picks up interrupted runs)
.venv\Scripts\python.exe -m vabs_gpt train --resume vabs_gpt\big.bin --steps 30000

.venv\Scripts\python.exe -m vabs_gpt info vabs_gpt\big.bin   # shape, step, val loss, options

# One full pass over train.bin (sets --steps from the data size and batch)
.venv\Scripts\python.exe -m vabs_gpt train --epochs 1 --out vabs_gpt\epoch.bin

# Score a checkpoint on all of val.bin (or the first --bytes of it)
.venv\Scripts\python.exe -m vabs_gpt eval vabs_gpt\epoch.bin
```

By default each training batch takes windows at random offsets. With
`--epochs`, train.bin is cut into non-overlapping windows and every window is
visited once per epoch, in a shuffled order that a resumed run continues. The
loss printed during training uses `--eval-batches` random validation batches;
`eval` uses a fixed prefix of val.bin, so it is the number to compare between
runs.

## Muon and MuonClip

```powershell
.venv\Scripts\python.exe -m vabs_gpt train --optimizer muonclip --lr 0.012 --batch 16 --beta2 0.99
```

`--optimizer muon` follows "Muon is Scalable for LLM Training" (Liu et al.
2025, `docs\muon\2502.16982v1.pdf`):

- **Which parameters:** the hidden weight matrices (q, k and v as three
  separate matrices, the attention output, both MLP layers) get Muon. The
  embeddings, the head, the biases and the layer norms stay on AdamW, with the
  same learning rate, schedule and weight decay.
- **The update:** Nesterov momentum `M = mu M + g`, orthogonalized from
  `g + mu M` by five quintic Newton-Schulz iterations (coefficients 3.4445,
  -4.7750, 2.0315), then `W -= lr (0.2 sqrt(max(rows, cols)) O + wd W)`. The
  0.2 sqrt(max(rows, cols)) factor gives the update roughly AdamW's RMS, which
  is why one learning rate serves both.
- **Settings:** `--muon-momentum` (0.95), `--muon-ns-steps` (5) and
  `--muon-scale` (0.2). Gradient clipping applies to the gradient before the
  momentum, as with AdamW.

`--optimizer muonclip` is Muon plus QK-Clip, the attention-logit control of
Kimi K2:

- **What it does:** the forward pass records every head's largest scaled
  attention logit `q.k / sqrt(head size)` in the step, over all micro-batches.
  After the update, each head whose maximum exceeded tau has its query and key
  weights and biases multiplied by `sqrt(tau / max)`. That scales its logits by
  exactly `tau / max`. Heads below tau are untouched.
- **tau:** `--qk-clip` sets tau (default 100, as in K2). It also works with
  `--optimizer adamw` or `muon`.
- **Logging:** with QK-Clip on, the log line shows the largest logit of the
  logged step and how many heads it clipped.

First comparison: default shape, lr 1.2e-2, batch 16, beta2 0.99, 3,000 steps
(12M bytes). Each run is scored with `eval` on the first 4M bytes of val.bin.

| Optimizer | Val loss | Largest logit at the end | Speed |
| --- | --- | --- | --- |
| AdamW | 1.044 | | 2.36M tok/s |
| Muon | **0.950** | 50.6 | 2.02M tok/s |
| MuonClip, tau 100 | 0.950 (never clipped) | 50.6 | 2.02M tok/s |
| MuonClip, tau 30 | 0.951 | 29.6 | 2.00M tok/s |

At this size the logits stay far below K2's tau of 100, so QK-Clip only
matters with a smaller tau or a larger model.

The speeds in the table are from the first version of the Muon kernels. The
optimized version below cut Muon's own GPU time per step by about 40% (default
shape, 0.27 to 0.16 ms); the same runs then trained at 2.2M tok/s, measured while
other programs were using the GPU.

How the Muon step was made faster (GPU time per step of the Newton-Schulz
products, default shape / the 1M-parameter shape):

- **Smaller workgroups.** The batched matrices are small: the default shape's
  64 x 256 batch is 2,048 invocations, only 8 workgroups of 256. With 64 per
  workgroup the X X^T product went from 0.157 to 0.084 ms.
- **The norm from trace(X X^T).** The first Newton-Schulz product of the
  unnormalized X has |X|^2 on its diagonal, so the momentum kernel is a plain
  per-element kernel and the first iteration scales by 1/|X| on the fly. The
  momentum kernel went from 0.034 to 0.010 ms (default shape).
- **Split-K.** X X^T sums over the long side of X. Chunks of 16 vec4s, run as
  separate workgroups and added up by `reduce_partials4`, gave 0.137 instead
  of 0.176 ms (default) and 0.30 instead of 0.37 ms (1M).
- **A transposed copy of X.** In X X^T both operands are rows of X, so
  neighbouring invocations read different rows and the loads are not
  coalesced. Each X update also writes X^T, which makes X X^T an ordinary
  product: 0.50 to 0.27 ms on the 1M shape.
- **The max-logit store last.** Storing each row's largest logit between
  `attn_fwd`'s two loops slowed it by 10%; at the end of the kernel, by 3%.

`bench --optimizer muon` (or `muonclip`) profiles these steps.

Muon's momentum is stored where AdamW keeps its first moment, so the
checkpoint format is unchanged. Resuming with a different optimizer family
(AdamW, or Muon/MuonClip) restarts the moments.

## 8-bit optimizer states

```powershell
.venv\Scripts\python.exe -m vabs_gpt train --optim-bits 8                           # quantized AdamW
.venv\Scripts\python.exe -m vabs_gpt train --optimizer muonclip --optim-bits 8 `
    --lr 0.012 --batch 16 --beta2 0.99                                               # quantized MuonClip
```

`--optim-bits 8` keeps the AdamW moments, and Muon's momentum, as 8-bit
blockwise states: the algorithm of bitsandbytes' 8-bit optimizers (Dettmers et
al., *8-bit Optimizers via Block-wise Quantization*, ICLR 2022), ported to vabs
in [`vabs_gpt_bitsandbytes`](../vabs_gpt_bitsandbytes) (MIT, Copyright (c)
Facebook, Inc. and its affiliates; see its `NOTICE.md`).

- **The state:** one byte per element, an index into bitsandbytes' dynamic
  quantization map (signed for the first moment and the momentum, unsigned for
  the second moment), plus a float32 absmax per block of 256 elements. Every
  tensor (each weight, bias and layer-norm half) has its own blocks. Muon's
  momentum is kept per batch of equally shaped matrices, in the order of the
  Newton-Schulz workspace.
- **The step:** dequantize, update in float32, find each block's new absmax,
  quantize again. AdamW follows bitsandbytes' kernel: the bias-corrected update,
  then decay as `W *= 1 - lr wd` on the weights. Muon's momentum kernel writes
  the Nesterov direction straight into the workspace.
- **Checkpoints and resuming:** a checkpoint keeps the codes and absmax values.
  The state's layout differs between AdamW alone and Muon/MuonClip, so the
  same rule as before applies: resuming with another optimizer family, or
  another `--optim-bits`, restarts the moments.
- **Differences from bitsandbytes:** small tensors are 8-bit too (bitsandbytes
  keeps tensors under 4,096 elements in 32-bit), NaN or infinite gradients are
  not skipped, and clipping uses the global norm computed on the GPU.

Same comparison as above (default shape, lr 1.2e-2, batch 16, beta2 0.99,
3,000 steps, `eval` on the first 4M bytes of val.bin):

| Optimizer | 32-bit states | 8-bit states |
| --- | --- | --- |
| AdamW, seed 0 | 1.044 | 1.012 |
| AdamW, seed 1 | 1.032 | 1.041 |
| MuonClip, tau 100 | 0.951 | 0.947 |

The two seeds show the noise at this budget, about 0.03: 8-bit states train as
well as 32-bit ones here. Speed is the same within noise (AdamW 2.51M and 2.44M
tok/s, MuonClip 2.25M both). By `bench` on the 1M-parameter shape, the 8-bit
updates take 0.024 ms of GPU time per step (AdamW, 0.020 at 32 bits) and 0.044
ms (AdamW + Muon momentum, 0.049 at 32 bits).

Optimizer state sizes (saved; on the device add about 30% of scratch for the
absmax copies and partial maxima):

| Shape | 32-bit | 8-bit AdamW | 8-bit Muon/MuonClip |
| --- | --- | --- | --- |
| default, 249K parameters | 2.0 MB | 0.52 MB | 0.32 MB |
| 1M parameters | 8.1 MB | 2.1 MB | 1.1 MB |

Checkpoints shrink from 3.0 MB to 1.5 MB (AdamW) and 1.3 MB (MuonClip) for the
default shape. At these sizes the activations still take most of the device
memory; the states matter for wide models with small batches.

## Hyperparameter sweeps

`sweep.py` trains every point of a grid for the same number of bytes, then
scores each one with `eval` on the same validation bytes:

```powershell
.venv\Scripts\python.exe -m vabs_gpt.sweep --dir vabs_gpt\sweep\lr --tokens 16e6 `
    --grid lr=1e-3,3e-3,5e-3 batch=16,32 --set weight-decay=0.1
.venv\Scripts\python.exe -m vabs_gpt.sweep --dir vabs_gpt\sweep\lr --report
```

Results go to `<dir>\results.jsonl`, with one training log per run.
Rerunning a command skips the points that already finished.

| Group | Options |
| --- | --- |
| Model | `--ctx` context bytes, `--dim`/`--embed` width, `--heads`, `--layers`/`--depth`, `--mlp` hidden width (default 4 × dim), `--init-std` |
| Optimization | `--steps` (number of optimizer steps, i.e. batches), `--epochs` (passes over train.bin; sets `--steps`), `--batch` (sequences per micro-batch), `--grad-accum` (micro-batches per step), `--lr`, `--min-lr`, `--schedule cosine\|linear\|constant`, `--warmup`, `--beta1`, `--beta2`, `--eps`, `--weight-decay`, `--grad-clip`, `--seed` |
| Optimizer | `--optimizer adamw\|muon\|muonclip`, `--optim-bits 32\|8` (see [8-bit optimizer states](#8-bit-optimizer-states)), `--muon-momentum`, `--muon-ns-steps`, `--muon-scale`, `--qk-clip` (tau; see [Muon and MuonClip](#muon-and-muonclip)) |
| Data and output | `--data`, `--out`, `--overwrite`, `--resume`, `--log-every`, `--eval-every`, `--eval-batches`, `--sample-every`, `--sample-len`, `--sample-prompt` |
| Config | `--config run.json`: any train option by its long name, e.g. `{"dim": 128, "layers": 6, "grad-accum": 4}` |

Rules and constraints:

- **Precedence:** command-line flags, then `--config`, then (with `--resume`)
  the options saved in the checkpoint, then the defaults.
- **Shape constraints:** `dim` must divide evenly into `heads`, and the head
  size, `mlp` and vocab must be multiples of 4 (the kernels use `vec4` loads).
  Any batch size and context length work. The trainer checks the estimated
  device memory before allocating.
- **Weight decay** is decoupled (AdamW). It applies to weight matrices and
  embeddings, not to biases or layer norms. Gradient clipping uses the global
  gradient norm, which the log prints every `--log-every` steps.
- **Checkpoint format:** a checkpoint holds the model shape, step, validation
  loss, the training options and the optimizer state (float32 Adam moments, or
  the 8-bit states with `--optim-bits 8`).
- **Resuming** keeps the checkpoint's shape and refuses conflicting shape
  flags. It saves back to the resumed file unless `--out` is given.
- **Overwriting:** `train` won't replace a different existing checkpoint
  without `--overwrite`.

## Sweep and full-epoch results

The model shape stayed at the ~250K-parameter budget. Every run was scored with
`eval` on the first 4M bytes of val.bin. The stage scripts, logs and
`results.jsonl` files are in `sweep\`.

1. **Learning rate × batch size** (16M bytes each). Smaller batches and higher
   rates won everywhere: batch 16 beat 32 and 64 at every rate.
2. **Higher rates and batch 8.** lr 1.2e-2 at batch 16 was best (1.037). Batch 8
   was no better and runs slower per byte.
3. **One change at a time** from that point. Changing the seed moved the loss
   by 0.007, which sets the noise level.
   - Helped: beta2 0.99 (0.997), warmup 500 (1.003), weight decay 0.1 (1.009).
   - Hurt: gradient clipping at 1.0 (1.059).
   - No effect: a lower final lr.
4. **Shapes with 228K–249K parameters.**
   - The best were the default shape with 8 heads (1.014) and 80 × 3 layers
     with MLP 224 (1.025, and 15% faster).
   - Deeper and narrower shapes were worse.
   - 96 × 2 layers stalled on a plateau at this learning rate.
5. **Combinations** of the stage-3 changes did not add up beyond noise.
6. **5× budget (82M bytes, as much as the original model.bin run).**

   | Settings | Val loss |
   | --- | --- |
   | **lr 1.2e-2, batch 16, beta2 0.99** | **0.844** |
   | + warmup 500, weight decay 0.1, 8 heads | 0.882 |
   | the same at lr 6e-3 | 0.909 |
   | the old defaults (lr 3e-3, batch 32) | 0.912 |

   The settings that won in short runs lost here, so the simplest change won.

Full epoch with the winner, 465,103 steps of 16 × 256 bytes (1.9 GB, each window
seen once), 4.3 hours:

```powershell
.venv\Scripts\python.exe -m vabs_gpt train --epochs 1 --lr 0.012 --batch 16 --beta2 0.99 `
    --out vabs_gpt\epoch.bin --eval-every 10000 --eval-batches 32 --log-every 2000
```

| Checkpoint | Training bytes | Loss on all of val.bin |
| --- | --- | --- |
| `model.bin` (defaults, 10,000 steps) | 82M | 0.9108 (1.314 bits/byte) |
| `epoch.bin` (sweep winner, 1 epoch) | 1.9G | **0.7525 (1.086 bits/byte)** |

The final training loss (0.753) matches the validation loss, so a single
epoch does not overfit. The log is `epoch.log`.

## How it works

- `gpu.py`: a headless `ComputeContext` with no window, surface or swapchain,
  providing what vabs' `Arena`, `ComputePipeline` and `Frame` need (the same
  approach as `pbd_gpu`). It also has `Recording`, which records a vabs `Frame`
  into a command buffer once so it can be resubmitted.
- `kernels.py`: the compute shaders:
  - embedding and layer norm;
  - linear layers (forward, input grad, weight grad);
  - causal attention (softmax, the score gradient, and two products with the
    attention matrix that serve the output and the q/k/v gradients; for head
    sizes up to 16 one kernel does the softmax and the output together);
  - cross-entropy;
  - chunked reductions, the squared gradient norm, AdamW, and a gather that
    builds the transposed weight copy;
  - Muon: a momentum kernel that gathers each matrix into a workspace (transposed
    where it has more rows than columns, so each is m x n with m <= n), a
    batched 4x4-tiled product for Newton-Schulz on all matrices of one shape at
    once, and a kernel that scatters the update back;
  - QK-Clip: two reductions to each head's largest logit and the rescaling of
    q and k.

  The DSL has no workgroup memory, so matrix products are register-tiled: one
  invocation computes a 4x4 block of outputs from `vec4` loads and `mat4`
  products. Weight gradients are summed over rows in two passes: partial sums
  per 128-row chunk (the last chunk may be shorter), then a reduction that
  adds into the gradient. That accumulation is what lets micro-batches add up.
  A 1-D dispatch wider than 32,768 workgroups continues in the y dimension,
  which keeps large batches and contexts under Vulkan's 65,535-group limit.
- `model.py`: `Model` keeps the parameters, gradients, Adam moments and the
  weight-decay mask as flat device-local arrays, plus a copy of the parameters
  with every linear layer transposed, which the forward pass reads. `Runner`
  allocates the activations for one batch size and records one command buffer
  per optimizer step: zero the gradients, forward, loss, backward, then the
  update (gradient norm, clipping, AdamW, refresh of the transposed copy).
  With `--grad-accum` there is one command buffer per micro-batch.

  Per-step values (tokens, learning rate, Adam bias corrections) go through
  host-visible buffers, so the recordings never change. Inference has no KV
  cache; it reruns the full window for each new byte.
- `test_gpu.py` checks the GPU against PyTorch:
  - the loss and every parameter's gradient, for the default shape and for
    odd ones (a short last chunk, row counts that are not multiples of 4, a
    non-4× MLP, head size 12);
  - a 2,048-byte context, and a dispatch wide enough to be split;
  - gradient accumulation against one large batch;
  - three AdamW steps with clipping and decay against a float64 reference;
  - three Muon steps (both matrix orientations, clipping, decay) and two
    MuonClip steps with gradient accumulation against a float64 reference, plus
    a check that QK-Clip brings every clipped head's largest logit to exactly
    tau;
  - 8-bit AdamW step by step against bitsandbytes' algorithm (from the GPU's
    own gradients and state), 8-bit MuonClip with gradient accumulation against a
    float64 reference, the 8-bit tensors' layout and decay flags, and resuming an
    8-bit checkpoint to bit-identical parameters;
  - option parsing.

  It needs torch, so it runs in `.ptvenv` with vabs' pure-Python source on the
  path:

  ```powershell
  $env:PYTHONPATH = "vabs\src"; .ptvenv\Scripts\python.exe -m unittest vabs_gpt.test_gpu -v
  ```

## Performance

On an RTX 4090, for the 1M-parameter shape (`--dim 128 --layers 5 --heads 8
--mlp 448`, batch 32 x 256 bytes), a training step takes about 5.4 ms
(1.5M bytes/s). The same step in PyTorch (`torch_ref.py`, float32, no
compilation) takes about 6.4 ms. The first version of these kernels took 23 ms.

```powershell
.venv\Scripts\python.exe -m vabs_gpt.bench            # ms/step and GPU time per kernel
$env:PYTHONPATH = "vabs\src"
.ptvenv\Scripts\python.exe -m vabs_gpt.compare        # vabs and PyTorch steps side by side
```

`bench` times every dispatch with GPU timestamp queries and keeps the fastest
sample of each, so it stays usable while something else is using the GPU.
`compare` alternates short bursts of both programs. What made the difference,
largest first:

- **Register tiling.** One invocation per output element loads two values per
  multiply-add. A 4x4 tile loads 8 `vec4`s for 64 multiply-adds.
- **Every invocation of a workgroup on the same rows.** The forward pass reads
  a transposed copy of the weights, and the transposed attention products walk
  the queries from the last one down, so neighbouring invocations load the same
  or adjacent addresses at each step.
- **Fewer passes over the attention matrix.** It is 67 MB per layer here, and
  the attention kernels are limited by memory bandwidth. The softmax computes
  its scores twice instead of storing and re-reading them, and the score
  gradient uses `sum(att * d_att) = dout . y` instead of a pass to add it up.
- **Independent dispatches without barriers between them.** The backward pass
  is recorded as stages (for example a layer's input gradient, its
  weight-gradient partial sums and the previous stage's reduction), and the
  GPU runs the dispatches of a stage side by side.
- **One submission per step**, with the per-token losses added up on the GPU.
- **Steps queued one ahead.** `Runner.submit_step` queues a step and returns
  the loss of the one before, so the GPU already has the next step while the
  host reads results and prepares a batch. Two sets of host-visible buffers
  (tokens, losses, step scalars) alternate, each with its own recording.
  `train` uses it and collects the outstanding step before every log line,
  evaluation, sample and save, so the output is unchanged. `train_step` still
  waits for its own step.
- **Sizes compiled into the forward kernels.** `Device.pipeline` builds the
  forward matmul, attention and layernorm kernels once per shape with
  `CompiledShader.specialize`, which turns the width and row-count push
  constants into SPIR-V constants so the driver can unroll the loops. The
  matmul and attention run 20-25% faster, the layernorm 8%. The same change
  made the backward kernels no faster (the matmul gradients 6% slower), so
  they stay generic.

- **Softmax and attention output in one kernel** for head sizes up to 16. In
  training it is about 13% faster than the two separate kernels; evaluation
  and sampling also skip storing the attention matrix and run about 20%
  faster.

Things that measured slower: 4x8 tiles, explicit multiply-add chains in place
of `mat4` products, workgroup sizes other than 256 for the tiled kernels, and
fusing the attention score gradient with the query gradient.
