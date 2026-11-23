# vabs_gpt.chat: a small Python chatbot on vabs compute

An implementation of `docs/small_python_chatbot.md`: a 33M-parameter decoder,
pretrained on a mix of Python code and educational English and fine-tuned on
short chats, trained and sampled entirely through vabs compute shaders. It
extends [`vabs_gpt`](..) (the TinyStories trainer there is unchanged) and keeps
its optimizer states in 8 bits with the bitsandbytes port in
[`vabs_gpt_bitsandbytes`](../../vabs_gpt_bitsandbytes).

## Results

All runs on one RTX 4090 (shared with a game for part of the baseline).

| Stage | Tokens / data | Time | Held-out result |
| --- | --- | ---: | --- |
| LR sweep | 16M tokens per run, 6 runs | 6 x 6 min | Muon beats AdamW by ~0.4 nats; Muon 4e-3 best, 8e-3 worse |
| Pilot | 50M tokens, Muon 3e-3 | 18 min | val stack 2.32, fineweb 4.07, cosmo 3.44; body pass@1 1.0% |
| Baseline | 500M tokens (15,258 steps of 32K) | 3.8 h | val stack **1.85**, fineweb **3.52**, cosmo **2.73**; body pass@1 **3.0%** |
| Chat fine-tune | 7,596 chats x 3 epochs (1.46M scored tokens, 489 steps) | 4 min | chat val loss 2.00 -> **1.03**; see below |

Per byte (validation bytes/token 3.40, 4.43, 4.98) the baseline is at 0.79
bits/byte on Python, 1.15 on FineWeb-Edu and 0.80 on Cosmopedia. Validation
loss was still falling ~0.01 per 1,000 steps at the end.

**Checkpoints of the baseline on the same data and prompts** (`eval`;
loss in nats/token, "chats" is the loss on the assistant tokens of the
held-out fine-tuning chats; code checks over 64 completions of the 16 function
prompts at temperature 0.6):

| step | cosmo | fineweb | stack | chats | finished | parses | brackets | indent |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 3.645 | 4.180 | 2.402 | 2.673 | 67% | 64% | 95% | 100% |
| 4,000 | 3.396 | 3.993 | 2.225 | 2.452 | 58% | 52% | 91% | 98% |
| 6,000 | 3.261 | 3.892 | 2.133 | 2.323 | 61% | 59% | 95% | 100% |
| 8,000 | 3.141 | 3.801 | 2.057 | 2.235 | 56% | 56% | 97% | 100% |
| 10,000 | 3.027 | 3.711 | 1.976 | 2.177 | 72% | 72% | 98% | 98% |
| 12,000 | 2.918 | 3.627 | 1.908 | 2.102 | 72% | 72% | 92% | 100% |
| 14,000 | 2.838 | 3.564 | 1.856 | 2.047 | 75% | 70% | 95% | 100% |
| 15,258 | 2.814 | 3.545 | 1.841 | 2.003 | 78% | 78% | 97% | 98% |

**Functional evaluation** (`funceval`, 200 held-out KodCode tasks not used by
any chat, greedy decoding, the task's own tests decide):

| | base (500M) | chat (fine-tuned) |
| --- | ---: | ---: |
| `body`: write the body from signature + docstring | 3.0% (6/200) | **27.0%** (54/200) |
| `chat`: answer the task's question | – | **18.5%** (37/200) |
| `fix`: fix the injected bug, given its real error | 9.0% (16/177) | **71.2%** (126/177) |

None of the buggy versions passes its tests, and replies that copy the bug
back unchanged (24 of 177) count as failures. Fix rate by the error shown:
`NameError` 79/121, `SyntaxError` 21/24, `ValueError` 15/19, a wrong return
value (off-by-one, missing `return`) 10/11.

Fine-tuning costs the pretraining distribution: validation loss on stack rose
1.84 -> 2.30 and on fineweb 3.54 -> 3.94 (3 epochs with Muon at lr 3e-4); the
chat loss was flat over the last third of the run, so one or two epochs or a
lower rate would likely keep more of it at little cost.

A greedy conversation with `chat.bin`:

```
you> Write a Python function that checks whether a string is a palindrome.
bot> Here's one way to write it:

    def is_palindrome(s):
        """
        Checks if the given string s is a palindrome.
        ...
        """
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned_s = ''.join(char.lower() for char in s if char.isalnum())
        # Check if the cleaned string is equal to its reverse
        return cleaned_s == cleaned_s[::-1]

For example, calling `is_palindrome("A man a plan a canal Panama")` gives `True`.

you> Now add a few tests for it.
bot> Here are some tests. Save the function as `solution.py` and run `pytest` in the same folder:
    (three copies of the same test, under the same name)

you> What is the capital of France?
bot> The capital of France is Paris.
```

Where it is weak: the follow-up tests repeat themselves; explanations of
errors in code it was not trained on (`items[len(items)]`) echo the code
instead of fixing it; usage examples sometimes quote a wrong result; and
general-knowledge answers beyond one sentence wander. This is a 33M model
after 500M tokens.

## The model

| | |
| --- | --- |
| Parameters | **33,055,232** (embedding 10.24M, 8 blocks of 2.85M, final norm) |
| Shape | 8 layers, width 512, SwiGLU hidden width 1,344 |
| Attention | 8 query heads, 4 key/value heads (grouped-query), head size 64, RoPE (base 10,000, interleaved pairs) |
| Norms | RMSNorm (pre-norm, a gain, no bias); no biases anywhere |
| Vocabulary | 20,000-token byte-level BPE; input embedding tied to the output head |
| Context | 1,024 tokens |

`model.py` follows `vabs_gpt.model`: parameters, gradients and optimizer state
are flat device arrays, the forward pass reads a copy with every matrix
transposed (the head reads the transposed embedding), and a `ChatRunner` records
a whole optimizer step (zero, forward, loss, backward, update) as one command
buffer, queued one step ahead. The new kernels are in `kernels.py`:

- `uint32` token ids, and the embedding gradient as a scatter: the host groups
  the batch's positions by token id, and one invocation per (distinct id, four
  channels) adds up its rows. (A dense pass over the 20,000-row table for every
  token would be 2,500x the work.)
- RMSNorm forward and backward; bias-free matrix products (4x4 register tiles,
  as in vabs_gpt); RoPE applied in place to q and k after the qkv product, and
  its inverse rotation to their gradients;
- grouped-query attention: softmax rows per query head, `att v` and `datt k`
  reading the shared key/value head, `att^T dout` and `datt^T q` summing over
  the query heads of a group;
- SwiGLU (`silu(gate) * up`) forward and backward;
- cross entropy over the large vocabulary in three dispatches (per 1,024-logit
  chunk: maximum and sum of exponentials; per row: log-sum-exp; per element:
  the gradient), with an ignored target id for masked tokens. The loss of a
  step is the mean over its scored tokens.

Weight gradients are summed in chunks of rows; each layer's chunk size is
chosen so its partial sums fit in a 128 MB scratch buffer (the tied
embedding's gradient is 10M floats). AdamW, the gradient norm, clipping, the
8-bit states and Muon's Newton-Schulz step are vabs_gpt's and
vabs_gpt_bitsandbytes' kernels, unchanged.

Speed on an RTX 4090: batch 8 x 1,024 tokens, AdamW with 8-bit states:
**162 ms per micro-batch, about 50,000 tokens/s**; Muon: 42,600–47,000 tokens/s.
Inference has no KV cache: each new token reruns the shortest window
(128, 256, 512 or 1,024 tokens) that holds the sequence, which RoPE allows
because positions start at 0 in every sequence.

### bfloat16 tensor cores (`--precision bf16`)

Every command that runs the model takes `--precision f32|bf16` (default f32).
With bf16 the matrix products run on the tensor cores through vabs'
cooperative matrices (VK_KHR_cooperative_matrix with VK_KHR_shader_bfloat16):
operands rounded to bfloat16, accumulation in float32. Parameters, activations,
gradients and optimizer states stay float32, so checkpoints are the same in
both modes and a run can switch precision when resumed.

One kernel, `mm_coop` in `kernels.py`, does all of it. Each 64 x 64 output tile
stages float32 slices of its operands into workgroup memory as bfloat16, and
four subgroups multiply them as 16 x 16 cooperative matrices. Layout flags,
batching (with a per-head mapping for grouped-query attention), split
reductions, causal limits and a scaled epilogue cover:

- the three products of every linear layer (forward, input and weight
  gradients), including the tied head;
- attention: the scores `q k^T` (tiles above the diagonal skipped), `att v`,
  and in the backward pass `dout v^T`, `ds k`, and `ds^T q` and `att^T dout`
  summed over each key/value head's query heads. Two row kernels
  (`attn_softmax_rows`, `attn_bwd_rows`) do the softmax and its gradient;
- Muon's Newton-Schulz iteration (X^T is X read column-major, so the
  transposed copy goes unused).

Products whose sizes are not multiples of 16 (small decoding batches, odd
windows) fall back to the float32 kernels. `ChatModel(..., precision='bf16')`
selects the mode in code, and `Device(ctx, precision=...)` sets the default of
the models made on it.

GPU time of one training step (`python -m vabs_gpt.chat.bench --precision ...`;
batch 8 x 1,024, Muon, 8-bit states; RTX 4090, minimum per dispatch over
repeated runs with another program using about 40% of the GPU):

| | float32 | bfloat16 |
| --- | ---: | ---: |
| 33M model | 209 ms (39k tok/s) | 61 ms (133k tok/s) |
| 66M model (width 640, 12 layers) | 473 ms (17k tok/s) | 118 ms (69k tok/s) |

`python -m vabs_gpt.chat.bench_mm` compares the linear-layer products alone:
64-76 TFLOP/s with bf16 against 7-27 TFLOP/s for the float32 kernels. In a
120-step pretraining run from the same initialization, the bf16 loss tracked
the float32 one to within 0.006 (5.860 against 5.855 at the end; validation
losses within 0.01), at 80k against 27k tok/s of wall-clock time while the GPU
was shared.

## Data

| Source | Share | Training tokens | Validation | Bytes/token |
| --- | ---: | ---: | --- | ---: |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu), Python | 40% | 200M | 2M, held-out repositories | 3.42 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 50% | 250M | 2M | 4.32 |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | 10% | 50M | 1M | 4.77 |

- **Stack-Edu** lists file ids only; `download.py` fetches the contents from
  Software Heritage's public S3 bucket (420,000 files of at most 8 KB from the
  first of five Python shards: complete short files). Files are kept whole and
  newlines normalized. Validation holds out every file of 1% of the
  repositories (by a hash of the repository name), so nearby files of one
  project stay on the same side.
- **FineWeb-Edu**: one shard of `sample/10BT`. **Cosmopedia**: shards of
  khanacademy, openstax, wikihow, stanford and web_samples_v2, read in turn,
  keeping only documents of at most 1,023 tokens ("short explanations that fit
  the context window").
- Documents end with `<|endoftext|>`. The shares are measured after
  tokenization: each source is tokenized up to its share of the 500M tokens.
- **Sampling**: each source is cut into non-overlapping 1,024-token windows
  visited in a shuffled order; sources take turns by smooth weighted
  round-robin, so every batch holds the mix exactly. `--mix` changes the
  weights without re-tokenizing; the sampler's state is saved in checkpoints.

**Tokenizer** (`tokenizer.py`, pure Python): byte-level BPE trained on an 80 MB
sample of the mix (40/50/10 by bytes), 19,740 merges in about 16 s. The
pretokenizer is GPT-4's with changes for code: identifiers (letters, digits,
underscores) stay whole, a run of spaces gives its last space to the next word
so indentation is its own token, punctuation keeps a following newline.
`prepare check` prints how it splits a function, a traceback and a Markdown
code block:

```
def| mean|(values|):
|   | """|Return| the| arithmetic| mean| of| a| list| of| numbers|."""
|   | if| not| values|:
|       | raise| ValueError|("|mean|()| of| an| empty| list|")

T|race|back| (|most| recent| call| last|):
| | File| "|main|.py|",| line| 12|,| in| <|module|>
|   | print|(items|[|3|])
|Index|Error|:| list| index| out| of| range|
```

Special tokens: `<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|end|>`. They
never come out of `encode`, so text a user types cannot forge a turn.

## Fine-tuning chats

`sft_data.py` builds 8,000 chats: two thirds coding, one third general.

| Kind | Share | What |
| --- | ---: | --- |
| write | 27% | a short task from [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1)'s "Filter" subset, answered with its solution and a usage example from its tests |
| complete | 10% | a function stub from KodCode's "Prefill" subset, completed |
| error | 17% | the solution with one injected bug and the error it really raises (or the `None` it returns), answered with an explanation and the fix |
| followup | 13% | a `write` chat, then an edit: tests, a usage example, a rename, or no docstring |
| general | 33% | [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), English, no code, no system prompt |

Checked answers: every solution passes its task's tests before it is used;
every quoted traceback, return value and example is the real output of running
the code shown; every edited version is run again. Tasks that touch files,
processes, the network, time or randomness are left out. The bugs: a
misspelled variable (`NameError`), an off-by-one `range(len(x) + 1)`
(`IndexError`), a missing colon (`SyntaxError`), a missing `return`.

Chats are laid out as
`<|endoftext|><|user|>...<|end|><|assistant|>...<|end|>...` and packed into
1,024-token windows without splitting any chat. The loss covers the assistant
replies and their `<|end|>` only.

## Reinforcement learning with verifiable rewards

An implementation of `docs/small_python_rlvr.md`: the model gets a function
stub (imports, signature, a docstring with the task and one usage example)
and writes the body; the reward is 1 when every hidden test passes, else 0.

**Results** (greedy pass@1 on held-out tasks; T=0.8 rows from 8 samples per
task, the unbiased estimator):

| | base | chat | sft (RL start) | **RL** (`r1`, 150 steps) | RL, 300 steps at 3x lr |
| --- | ---: | ---: | ---: | ---: | ---: |
| MBPP test (474 tasks, all 3 tests) | 1.5% | 4.0% | 4.9% | **6.1%** | 6.3% |
| MBPP+ (217 of those tasks, EvalPlus inputs) | 0.9% | 5.1% | 4.6% | **6.0%** | 5.5% |
| KodCode `funceval body` (200 tasks) | 3.0% | 27.0% | 30.0% | **34.0%** | 33.5% |
| MBPP test pass@1 at T=0.8 | | | 2.3% | **4.5%** | |
| MBPP test pass@8 at T=0.8 | | | 9.3% | **10.1%** | |

- **The SFT step.** `sft.bin` is `base.bin` fine-tuned (2 epochs, Muon 3e-4,
  3 minutes) on the 7,596 chats plus 5,761 stub -> body completions from
  KodCode-V1 tasks that are in no RL or evaluation set. Sampling 8 completions
  of 1,000 training tasks at T=1, the share of tasks that pass *sometimes* (the
  ones that give GRPO a signal) is 4% for `base.bin`, 20% for `chat.bin` and
  29% for `sft.bin`. MBPP gives little signal (11 of 93 tasks); most comes
  from KodCode-Light. It also forgets less pretraining than `chat.bin` (val
  stack 2.19 vs 2.30).
- **RL** (`r1`): 150 steps of 32 tasks x 8 samples, AdamW lr 1e-5, 18 minutes.
  Training reward (25-step means) 0.12 -> 0.17; 17 MBPP test tasks gained, 11
  lost. Bodies became much shorter (median 110 -> 30 characters: the policy
  prefers one `return` expression), and sampled answers improved more than
  greedy ones. At 3x the rate for 300 steps the training reward kept rising
  (0.23) while held-out results did not: it was fitting the training tasks.

**How it works**

- `rl_data.py`: MBPP's train split (ids 601-974) plus 2,985 easy single-function
  KodCode-Light-RL-10K tasks for training, MBPP validation (511-600) as dev,
  MBPP test (11-510) and its MBPP+ versions held out. The reward uses only
  the tests the prompt does not show. Every reference passes in the sandbox;
  MBPP tasks whose tests need anything besides the one function (a `Pair`
  class next to it, say) are dropped, since a body cannot supply it.
- `verify.py`, the sandbox: a fresh base interpreter (`-I -B -S`, empty
  environment) per check in an empty temporary directory; a Windows job
  object (512 MB, one process so nothing can be spawned, killed on close; or
  `setrlimit` elsewhere) and a 5 s wall clock (Windows enforces job CPU limits
  only seconds late); an audit hook installed before the code is imported
  blocks network/process/`ctypes` modules, `os.system`/`exec`/`spawn`/`kill`,
  and every file write, delete or rename; the runner prints a random nonce
  only after all tests pass, so printed text or `exit(0)` cannot fake a pass.
  28 workers check about 100 completions per second.
- `decode.py`, fast sampling: a prefill forward over the prompts, whose
  per-layer qkv buffers then serve as the key/value cache, and one decode step
  per token (qkv of the new token, RoPE at its position, attention over its own
  rows), 8 steps per submission. Tokens are drawn by Gumbel-max on the GPU from
  the full softmax, so only ids come back: about 26,000 tokens/s at batch 128,
  against a few hundred for the window-rerunning `Generator`. Greedy output is
  token-for-token identical to it.
- `rlvr.py train`, GRPO without a reference model or KL term: advantages
  `(r - mean) / std` within each task's group (`--no-std` drops the division),
  groups with one reward left out, and one optimizer step per batch of samples,
  so the update is exactly on-policy and needs no clipped ratio. The loss is
  `sum A (-log p) / tokens` over the sampled tokens, a cross entropy with a
  weight per token (`ChatRunner(weighted=True)`; the weights may be negative).
  A step is about 5-9 s: sampling 2 s, checking 2-6 s, training the rest.

## Scaling up: 66M parameters, 1B tokens

The same recipe at twice the parameters and twice the data: `--layers 12 --dim
640 --heads 10 --kv-heads 5 --mlp 1664` (65,900,160 parameters, 20.6 GiB) on
1B tokens of the same 40/50/10 mix (`prepare tokens --tokens 1e9 --out-dir
data/1b`; the validation files are the same and the 500M set is a prefix of
it). Muon 3e-3 again: a 16M-token sweep put the best rate at 4e-3 for both
sizes, with 8e-3 clearly worse. Pretraining took 18.3 h (30,517 steps of 32K
at 20-22k tok/s, slowed to 7-9k tok/s while the GPU was shared with a game).
Every later step used the 33M commands unchanged, with outputs in `runs/m66/`
(`runs/m66/pipeline.sh`).

| | 33M, 500M tokens | **66M, 1B tokens** |
| --- | ---: | ---: |
| val stack / fineweb / cosmo (nats/token) | 1.841 / 3.545 / 2.814 | **1.673 / 3.316 / 2.524** |
| val held-out chats (base model) | 2.003 | **1.865** |
| chat fine-tune: chat val loss | 1.034 | **0.930** |
| `funceval body` pass@1, base | 3.0% | **9.5%** |
| `funceval body`, chat | 27.0% | **34.5%** |
| `funceval chat`, chat | 18.5% | **24.0%** |
| `funceval fix`, chat | 71% | **81.9%** |
| `funceval body`, sft / r1 | 30.0% / 34.0% | **43.0% / 41.5%** |
| MBPP test, base / chat / sft / r1 | 1.5 / 4.0 / 4.9 / 6.1% | 1.5 / **7.6 / 8.0 / 9.1%** |
| MBPP+, base / chat / sft / r1 | 0.9 / 5.1 / 4.6 / 6.0% | **2.3 / 9.7 / 8.3 / 11.1%** |
| AceCode held-out (500 tasks, ~20 asserts), base / chat / sft / r1 | 3.2 / 7.4 / 11.0 / 13.6% | **5.6 / 12.4 / 14.6 / 17.0%** |
| RLVR probe of sft: tasks that sometimes pass | 29% | **38%** |
| r1 dev pass@1, step 0 -> 150 | 6.0 -> 9.5% | **10.7 -> 15.5%** |

**Loss at the same token count** (`eval`, val stack). With twice as many
steps the 66M's cosine schedule is still near peak lr when the 33M is decaying,
so the bigger model leads early, falls behind in the middle and only wins once
its own lr comes down:

| tokens | 131M | 262M | 393M | ~500M | 1B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 33M | 2.225 | 2.057 | 1.908 | **1.841** (final) | |
| 66M | **2.166** | **2.044** | 1.972 | 1.895 (step 16,000) | **1.673** |

- The 66M passed the 33M's final loss between 590M and 655M tokens (steps 18,000-20,000); per byte it ends
  at 0.71 bits/byte on Python (33M: 0.78).
- **Fine-tuning gains carry over.** Every downstream number improves, most
  where the 33M was weakest: base body pass@1 triples, `fix` +11 points, and
  RL starts from far more signal (38% of probe tasks mixed vs 29%).
- **RL helps MBPP but not funceval here**: r1 lifts MBPP test 8.0 -> 9.1% and
  MBPP+ 8.3 -> 11.1%, but `funceval body` drops 43.0 -> 41.5% (the 33M went
  30 -> 34%). Bodies again shrink (mean length about 60 -> 32 tokens), and the
  KodCode-style funceval tasks reward the longer SFT bodies.
- SFT forgetting is about the same in absolute terms (val stack 1.67 -> 2.19
  after the chat fine-tune, 2.03 after sft) as at 33M (1.84 -> 2.30 / 2.19).

### AceCode-87K post-training (66M)

AceCode-87K (TIGER-Lab) has synthesized asserts, about 20 per task, against
MBPP's 3. It has no evaluation split, so `rl_data acecode` holds out 500 tasks
(`rl_acecode_eval`), takes 5,000 RL tasks and 5,000 SFT bodies (`--sft`) from
the rest, and keeps the three sets free of shared function names (see
`acecode_task`). Two variants, both with the same RL as `r1` (150 steps, AdamW
1e-5) but on MBPP + KodCode + AceCode (`--split train acecode`, 8,335 tasks):

- `ace`: RL from `sft.bin`;
- `ace2`: RL from `sft2.bin` = `base.bin` fine-tuned on `bodies2` (the `sft`
  data plus the 5,000 AceCode bodies) for 1 epoch instead of 2, with 2 of the
  8 rows of every micro-batch replaying pretraining data (`--pretrain-rows 2
  --data data/1b`), against forgetting.

Greedy pass@1 (`runs/m66/pipeline2.sh`):

| | sft | r1 | **ace** | sft2 | **ace2** |
| --- | ---: | ---: | ---: | ---: | ---: |
| MBPP test (474) | 8.0% | 9.1% | **10.1%** | 8.6% | 8.6% |
| MBPP+ (217) | 8.3% | 11.1% | 10.1% | 9.2% | **12.4%** |
| AceCode held-out (500) | 14.6% | 17.0% | 23.8% | 25.6% | **30.2%** |
| `funceval body` (200 KodCode) | **43.0%** | 41.5% | 40.5% | 38.0% | 31.5% |
| val stack after SFT (base 1.67) | 2.02 | | | **1.72** | 1.74 |
| RL: tasks with signal, step 1 -> 150 | | 19 -> 22% | 28 -> 38% | | 25 -> 41% |

- **Replay fixes forgetting**: two replay rows in eight keep val stack at 1.72
  (base 1.67) instead of 2.02, and fineweb/cosmo likewise (3.37 / 2.58 vs
  3.62 / 2.83), for a slightly worse chat loss (0.94 vs 0.90).
- **AceCode tasks give RL far more signal** (38-41% of groups mixed at the end
  vs 22% for `r1`) and the AceCode score rises most, though for `sft2`/`ace2`
  part of that is in-distribution SFT (`sft2` alone is at 25.6%).
- **Neither variant dominates.** `ace` is the most even (best MBPP test, a
  small loss on funceval); `ace2` is best on AceCode and MBPP+ but loses 11.5
  points of funceval. Bodies shrink further (RL mean length 47 -> 23 tokens) and
  `sft2` saw the KodCode bodies once instead of twice, both of which hurt the
  KodCode-style funceval. MBPP differences of 1-2 points are about one standard
  error (474 tasks), the AceCode and funceval gaps are not.

## Run

From the workspace root. Data preparation needs `pyarrow` (`.ptvenv`); training
and sampling need only vabs (`.venv`).

```powershell
# 1. Data (about 4.5 GB of downloads; Stack-Edu runs at ~500 files/s)
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.download
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.prepare tokenizer     # + prints the checks
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.prepare tokens --tokens 500e6
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.sft_data              # chats_train/val.jsonl
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.funceval build        # held-out coding tasks

# 2. Pilot (50M tokens), then the 500M-token baseline
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m vabs_gpt.chat pretrain --tokens 50e6 --optimizer muon --lr 3e-3 `
    --out vabs_gpt\chat\runs\pilot.bin
.venv\Scripts\python.exe -m vabs_gpt.chat pretrain --tokens 500e6 --optimizer muon --lr 3e-3 `
    --save-every 2000 --out vabs_gpt\chat\runs\base.bin

# 3. Fine-tune and talk
.venv\Scripts\python.exe -m vabs_gpt.chat finetune --init vabs_gpt\chat\runs\base.bin `
    --optimizer muon --lr 3e-4 --out vabs_gpt\chat\runs\chat.bin
.venv\Scripts\python.exe -m vabs_gpt.chat chat vabs_gpt\chat\runs\chat.bin

# Evaluate checkpoints on the same data and prompts
.venv\Scripts\python.exe -m vabs_gpt.chat eval vabs_gpt\chat\runs\base_step2000.bin vabs_gpt\chat\runs\base.bin
.venv\Scripts\python.exe -m vabs_gpt.chat.funceval run vabs_gpt\chat\runs\base.bin --mode body
.venv\Scripts\python.exe -m vabs_gpt.chat.funceval run vabs_gpt\chat\runs\chat.bin --mode chat

# 4. RLVR: tasks, the stub -> body fine-tune, then GRPO
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.rl_data download
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.rl_data build        # rl_train/dev/test/plus.jsonl
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.rl_data sft          # bodies_train/val.jsonl
.venv\Scripts\python.exe -m vabs_gpt.chat finetune --init vabs_gpt\chat\runs\base.bin `
    --optimizer muon --lr 3e-4 --epochs 2 --chats vabs_gpt\chat\data\bodies_train.jsonl `
    --val-chats vabs_gpt\chat\data\bodies_val.jsonl --out vabs_gpt\chat\runs\sft.bin
.venv\Scripts\python.exe -m vabs_gpt.chat.rlvr probe vabs_gpt\chat\runs\sft.bin --limit 1000
.venv\Scripts\python.exe -m vabs_gpt.chat.rlvr train --init vabs_gpt\chat\runs\sft.bin `
    --out vabs_gpt\chat\runs\rl\r1.bin --steps 150 --lr 1e-5
.venv\Scripts\python.exe -m vabs_gpt.chat.rlvr eval vabs_gpt\chat\runs\sft.bin `
    vabs_gpt\chat\runs\rl\r1.bin --split test plus

# Optional: AceCode-87K training tasks (about 17 synthesized asserts each, 1 GB download)
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.rl_data download --acecode
.ptvenv\Scripts\python.exe -m vabs_gpt.chat.rl_data acecode --limit 5000 --min-pass 0.5
.venv\Scripts\python.exe -m vabs_gpt.chat.rlvr probe vabs_gpt\chat\runs\sft.bin --split acecode
.venv\Scripts\python.exe -m vabs_gpt.chat.rlvr train --init vabs_gpt\chat\runs\sft.bin `
    --split train acecode --out vabs_gpt\chat\runs\rl\ace.bin --steps 150 --lr 1e-5
```

Ctrl+C saves a checkpoint; `--resume` continues the run in `--out` (model,
optimizer state and data position). Every command has `--help`.

What training prints and logs (`<out>.jsonl`): the loss and gradient norm, the
validation loss **separately for Python (stack) and prose (fineweb, cosmo)**,
sample completions of a code and a prose prompt, and **code checks**: 16
function prompts are completed and scored for whether the function is
finished (it dedents back to the top level), parses, closes its brackets and
indents by multiples of four spaces. `funceval` runs held-out KodCode tasks'
tests on the model's code: `body` (write the body from a signature and
docstring), `chat` (answer the question), `fix` (fix an injected bug).

## Tests

```powershell
$env:PYTHONPATH = "vabs\src"; .ptvenv\Scripts\python.exe -m unittest vabs_gpt.chat.test_chat -v
```

Loss and every parameter gradient against a float64 PyTorch implementation of
the same model (grouped-query with 2 and 4 query heads per key/value head, and
without grouping; windows shorter than the context; masked targets;
multi-chunk cross entropy; gradient accumulation; long weight-gradient
chunks), evaluation and inference logits on a short window, AdamW and Muon
with 32- and 8-bit states reducing the loss, checkpoint round trips to
bit-identical parameters, freeing device memory, the tokenizer (round trips,
no special tokens from text, indentation tokens) and chat masking and packing.
For RLVR: per-token weighted gradients (both signs) against PyTorch, the
cached decoder's greedy picks against the full forward, the GPU sampler's
frequencies against the softmax, completion records, advantages and the
training-row layout, MBPP stubs, and the sandbox refusing wrong answers,
sockets, file writes, `os.system`, faked output and `exit(0)`.
For bf16: `mm_coop` against float64 products of bf16-rounded inputs (every
layout, ragged tiles, residuals, split reductions), gradients of models whose
products all run on the tensor cores (including a 144-token window, with
attention tiles above, on and below the diagonal), the float32 fallback for
other sizes, the cached decoder at 16 sequences, and a Muon step against the
float32 one (Newton-Schulz alone, and everything in bf16).
