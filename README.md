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
  loss, the training options and the Adam moments.
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
- `kernels.py`: 18 compute shaders:
  - embedding and layer norm;
  - linear layers (forward, input grad, weight grad);
  - causal attention (softmax, output, three backward kernels);
  - cross-entropy;
  - a chunked reduction, the squared gradient norm, and AdamW.

  Each invocation computes one output element or one row, because the DSL has
  no workgroup memory. Weight gradients are summed over rows in two passes:
  partial sums per 64-row chunk (the last chunk may be shorter), then a
  reduction that adds into the gradient. That accumulation is what lets
  micro-batches add up. A 1-D dispatch wider than 32,768 workgroups continues
  in the y dimension, which keeps large batches and contexts under Vulkan's
  65,535-group limit.
- `model.py`: `Model` keeps the parameters, gradients, Adam moments and the
  weight-decay mask as flat device-local arrays. `Runner` allocates the
  activations for one batch size and records three command buffers:
  1. zero the gradients;
  2. forward, loss and backward, submitted once per micro-batch;
  3. the update: gradient norm, clipping and AdamW.

  Per-step values (tokens, learning rate, Adam bias corrections) go through
  host-visible buffers, so the recordings never change. Inference has no KV
  cache; it reruns the full window for each new byte.
- `test_gpu.py` checks the GPU against PyTorch:
  - the loss and every parameter's gradient, for the default shape and for
    odd ones (a short last chunk, a non-4× MLP, head size 12);
  - a 2,048-byte context that forces split dispatches;
  - gradient accumulation against one large batch;
  - three AdamW steps with clipping and decay against a float64 reference;
  - option parsing.

  It needs torch, so it runs in `.ptvenv` with vabs' pure-Python source on the
  path:

  ```powershell
  $env:PYTHONPATH = "vabs\src"; .ptvenv\Scripts\python.exe -m unittest vabs_gpt.test_gpu -v
  ```

## Performance notes (RTX 3050 Ti Laptop)

With the defaults, a training step takes about 55–60 ms (about 150K bytes/s), so
10,000 steps take about 10 minutes. The first version took 178 ms of GPU time
per step; two layout changes brought it down. Contiguous dot products now load
`vec4`s, via `F4` views of the same buffers. The attention matrix is stored
transposed (`att[b, h, key, query]`), so per-query threads touch neighbouring
addresses.
