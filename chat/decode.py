"""Fast sampling: a key/value cache and sampling on the GPU.

``Decoder(model, B, T)`` runs a prefill forward over up to B prompts (an
inference ``ChatRunner`` of B x T tokens, whose per-layer qkv buffers then serve
as the key/value cache), then decodes one token per sequence per step: the new
token's qkv row is rotated at its position and stored in the cache, it attends
over its own sequence's rows, and the next token is picked by Gumbel-max
(``argmax logits / temperature + Gumbel noise``, which samples the softmax
exactly; greedy without the noise). Only the sampled ids come back to the host,
``steps`` decode steps per submission.

Sampling is from the full softmax at the given temperature (no top-k), which
is what reinforcement learning needs: the samples come from the policy whose
log-probabilities the update differentiates.
"""
import array
import random

from vabs.memory import MemoryUsage

from vabs_gpt import model as GM
from vabs_gpt.gpu import Recording, compute_barrier

from . import kernels as K
from .model import ChatRunner, _new_arenas, release_arenas
from .tokenizer import END, EOT

_ceil = GM._ceil


class Decoder:
    def __init__(self, model, B, T, steps=8):
        cfg = model.cfg
        self.model, self.dev, self.cfg = model, model.dev, cfg
        self.B, self.T, self.steps = B, T, steps
        self.prefill = ChatRunner(model, B, 'infer', T=T)
        C, V, H, F, S = cfg.dim, cfg.vocab, cfg.heads, cfg.mlp, cfg.qkv_dim
        self.chunks = _ceil(V // 4, K.CE_CHUNK)
        before = {id(a) for a in self.dev.ctx._arenas}
        sizes = [B * C, B * C, B * C, B * S, B * C, B * 2 * F, B * F, B, B * H * T, B * V,
                 2 * B * self.chunks]
        regs = self.dev.arena([4 * s for s in sizes], MemoryUsage.DEVICE_LOCAL)
        (self.x, self.res2, self.ln, self.qkv, self.y, self.pre, self.h, self.rstd, self.sc,
         self.logits, self.part) = (r.device_address for r in regs)
        host = self.dev.arena([4 * B, 4 * B, 8, 4, 4 * steps * B], MemoryUsage.HOST_VISIBLE)
        self.tok, self.pos, self.knobs, self.seed, self.hist = host
        self.rec = Recording(self.dev.ctx)
        with self.rec.record() as frame:
            compute_barrier(frame)
            for k in range(steps):
                self._step(frame, k)
        self._arenas = _new_arenas(self.dev.ctx, before)

    def release(self):
        self.prefill.release()
        release_arenas(self.dev.ctx, self._arenas, [self.rec])

    def _step(self, frame, k):
        m, cfg, B, T = self.model, self.cfg, self.B, self.T
        run = self.dev.run
        C, V, H, F, S = cfg.dim, cfg.vocab, cfg.heads, cfg.mlp, cfg.qkv_dim
        hs4, C4, KV4, S4 = cfg.head_size // 4, C // 4, cfg.kv_dim // 4, S // 4
        eps = cfg.norm_eps
        pos = self.pos.device_address
        run(frame, 'embed_fwd', K.EmbedPush(x=self.x, wte=m.p('wte'),
                                            toks=self.tok.device_address, N=B, C=C), B * C4)
        dec = dict(cs=self.prefill.rope, sc=self.sc, pos=pos, B=B, T=T, H=H,
                   G=H // cfg.kv_heads, hs4=hs4, S4=S4, C4=C4, KV4=KV4)
        for i, a in enumerate(self.prefill.layer):
            h = f'h{i}.'
            run(frame, 'rmsnorm_fwd', K.NormPush(out=self.ln, inp=self.x, w=m.p(h + 'n1'),
                                                 rstd=self.rstd, N=B, C=C, eps=eps), B)
            m.linear(frame, self.qkv, self.ln, h + 'qkv', B, C, S)
            run(frame, 'rope_store', K.DecodePush(x=self.qkv, cache=a['qkv'], **dec), B * S4)
            run(frame, 'decode_scores', K.DecodePush(x=self.qkv, cache=a['qkv'], **dec),
                B * H * T)
            run(frame, 'decode_av', K.DecodePush(x=self.y, cache=a['qkv'], **dec), B * H * hs4)
            m.linear(frame, self.res2, self.y, h + 'proj', B, C, C, res=self.x)
            run(frame, 'rmsnorm_fwd', K.NormPush(out=self.ln, inp=self.res2, w=m.p(h + 'n2'),
                                                 rstd=self.rstd, N=B, C=C, eps=eps), B)
            m.linear(frame, self.pre, self.ln, h + 'fc', B, C, 2 * F)
            run(frame, 'swiglu_fwd', K.GluPush(pre=self.pre, h=self.h, N=B, F=F), B * F // 4)
            m.linear(frame, self.x, self.h, h + 'down', B, F, C, res=self.res2)
        run(frame, 'rmsnorm_fwd', K.NormPush(out=self.ln, inp=self.x, w=m.p('nf'),
                                             rstd=self.rstd, N=B, C=C, eps=eps), B)
        m.linear(frame, self.logits, self.ln, 'wte', B, C, V)
        sp = K.SamplePush(logits=self.logits, part=self.part, knobs=self.knobs.device_address,
                          seed=self.seed.device_address, tok=self.tok.device_address, pos=pos,
                          hist=self.hist.device_address, B=B, V=V, chunks=self.chunks, T=T,
                          step_=k)
        run(frame, 'sample_partial', sp, B * self.chunks)
        run(frame, 'sample_finish', sp, B)

    def generate(self, prompts, max_new=200, temperature=1.0, stop=(END, EOT), rng=None,
                 stop_fn=None, keep_stop=False):
        """New ids of each prompt (without the stop id), as ``Generator.generate``.

        ``temperature`` 0 is greedy. A sequence also ends when it fills the window T.
        ``keep_stop`` keeps the stop id that ended a sequence (it was sampled too).
        """
        rng = rng or random.Random()
        B, T = self.B, self.T
        n = len(prompts)
        if not 0 < n <= B:
            raise ValueError(f'{n} prompts for a decoder of batch {B}')
        prompts = [list(p) for p in prompts] + [[EOT]] * (B - n)
        if max(len(p) for p in prompts) >= T:
            raise ValueError(f'a prompt of {max(len(p) for p in prompts)} tokens for T={T}')
        toks = array.array('I')
        for p in prompts:
            toks.extend(p[:-1] + [EOT] * (T - len(p) + 1))
        pf = self.prefill
        pf._write_batch(pf.slots[0].regions, toks, None)
        pf.pos.write(array.array('I', [0] * B).tobytes())
        pf.rec.submit()
        self.tok.write(array.array('I', [p[-1] for p in prompts]).tobytes())
        self.pos.write(array.array('I', [len(p) - 1 for p in prompts]).tobytes())
        greedy = temperature <= 0
        self.knobs.write(array.array('f', [1.0 if greedy else 1.0 / temperature,
                                           0.0 if greedy else 1.0]).tobytes())
        out = [[] for _ in range(n)]
        done = [False] * n
        length = [len(p) for p in prompts]
        stop = set(stop)
        while not all(done):
            self.seed.write(array.array('I', [rng.getrandbits(32)]).tobytes())
            self.rec.submit()
            hist = array.array('I', bytes(self.hist.map())[:4 * self.steps * B])
            for k in range(self.steps):
                for b in range(n):
                    if done[b]:
                        continue
                    t = hist[k * B + b]
                    if t in stop:
                        done[b] = True
                        if keep_stop:
                            out[b].append(t)
                        continue
                    out[b].append(t)
                    length[b] += 1
                    if (len(out[b]) >= max_new or length[b] >= T
                            or (stop_fn and stop_fn(b, out[b]))):
                        done[b] = True
        return out
