"""Compute shaders for a GPT-style transformer: forward, backward and Adam.

All tensors are flat float32 arrays addressed through ``BufferRef`` push
constants, row-major, with ``N = B*T`` token rows. Each kernel computes one
output element (or one row) per invocation; the shader language has no
workgroup memory, so there is no tiling. Weight gradients sum over the N
rows in two passes: a ``*_partial`` kernel writes one partial sum per chunk
of R rows (dispatch y = chunk), and ``reduce_partials`` adds the chunks.

A linear layer's parameters are ``W[OC, C]`` followed by ``b[OC]``; a layer
norm's are ``gamma[C]`` followed by ``beta[C]``. Partial-sum index ``j``
follows the same order, so one reduction writes the weight and bias grads.

Element-wise kernels index with ``flat_index``: a dispatch wider than
``SPLIT_GROUPS`` workgroups continues in global_id.y (``Device.run`` splits it),
which keeps large batches and contexts under the 65535-group dispatch limit.
Chunked kernels allow a short last chunk, so N need not be a multiple of R.
Weight gradients accumulate (``+=``) so micro-batches can be summed; the
trainer zeroes them before each optimizer step.

Tokens are bytes packed four per ``uint32`` word (little endian).
Shaders cannot read Python globals, so all sizes come in push constants.
Integer ``/`` in a shader is integer division (the DSL has no ``//``).
"""
from vabs.shaders import compute_shader, shader_fn, dot, exp, inversesqrt, log, max, min, sqrt
from vabs.structs import BufferRef, float32, gpu_struct, uint32, uvec3, vec4

WG = 256  # local size of every kernel below
SPLIT_GROUPS = 32768  # workgroups per dispatch row; flat_index hardcodes WG * this


@gpu_struct
class F:
    v: float32


@gpu_struct
class F4:
    v: vec4


@gpu_struct
class U:
    v: uint32


@gpu_struct
class Hyper:
    """Per-step scalars, written by the host before each submission."""
    lr: float32
    beta1: float32
    beta2: float32
    eps: float32
    bc1: float32  # 1 - beta1**step
    bc2: float32  # 1 - beta2**step
    wd: float32   # decoupled weight decay (AdamW), applied where the decay mask is 1
    clip: float32  # max global gradient norm; 0 disables clipping


@shader_fn
def flat_index(gid: uvec3) -> uint32:
    return gid.x + gid.y * uint32(8388608)  # WG * SPLIT_GROUPS


@shader_fn
def byte_at(buf: BufferRef[U], i: uint32) -> uint32:
    return (buf[i >> uint32(2)].v >> ((i & uint32(3)) * uint32(8))) & uint32(255)


@shader_fn
def tanh_(x: float32) -> float32:
    return 1.0 - 2.0 / (exp(2.0 * x) + 1.0)


@shader_fn
def gelu(x: float32) -> float32:
    return 0.5 * x * (1.0 + tanh_(0.7978845608 * (x + 0.044715 * x * x * x)))


@shader_fn
def gelu_grad(x: float32) -> float32:
    th = tanh_(0.7978845608 * (x + 0.044715 * x * x * x))
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * 0.7978845608 * (
        1.0 + 3.0 * 0.044715 * x * x)


# -- reductions ---------------------------------------------------------------

@gpu_struct
class ReducePush:
    out: BufferRef[F]
    part: BufferRef[F]
    M: uint32   # values per chunk
    K: uint32   # chunks
    accumulate: uint32  # add into out instead of overwriting


@compute_shader(local_size=(WG, 1, 1))
def reduce_partials(global_id: uvec3, p: ReducePush):
    j = flat_index(global_id)
    if j >= p.M:
        return
    acc = 0.0
    for k in range(p.K):
        acc += p.part[uint32(k) * p.M + j].v
    if p.accumulate != uint32(0):
        acc += p.out[j].v
    p.out[j].v = acc


# -- embeddings ---------------------------------------------------------------

@gpu_struct
class EmbedPush:
    x: BufferRef[F]      # [N, C] output (forward) or its gradient (backward)
    wte: BufferRef[F]    # [V, C] token table, or partial sums [K, V*C] (backward)
    wpe: BufferRef[F]    # [T, C] position table or its gradient
    toks: BufferRef[U]   # [N] packed bytes
    N: uint32
    T: uint32
    C: uint32
    M: uint32            # backward: V*C
    R: uint32            # backward: rows per chunk
    accumulate: uint32   # backward: add into the position-table gradient


@compute_shader(local_size=(WG, 1, 1))
def embed_fwd(global_id: uvec3, p: EmbedPush):
    i = flat_index(global_id)
    if i >= p.N * p.C:
        return
    n = i / p.C
    c = i % p.C
    p.x[i].v = p.wte[byte_at(p.toks, n) * p.C + c].v + p.wpe[(n % p.T) * p.C + c].v


@compute_shader(local_size=(WG, 1, 1))
def embed_bwd_wte_partial(global_id: uvec3, p: EmbedPush):
    j = global_id.x
    k = global_id.y
    if j >= p.M:
        return
    v = j / p.C
    c = j % p.C
    acc = 0.0
    for r in range(min(p.R, p.N - k * p.R)):
        n = k * p.R + uint32(r)
        if byte_at(p.toks, n) == v:
            acc += p.x[n * p.C + c].v
    p.wte[k * p.M + j].v = acc


@compute_shader(local_size=(WG, 1, 1))
def embed_bwd_wpe(global_id: uvec3, p: EmbedPush):
    j = flat_index(global_id)
    if j >= p.T * p.C:
        return
    acc = 0.0
    for b in range(p.N / p.T):
        acc += p.x[uint32(b) * p.T * p.C + j].v
    if p.accumulate != uint32(0):
        acc += p.wpe[j].v
    p.wpe[j].v = acc


# -- layer norm ---------------------------------------------------------------

@gpu_struct
class LNPush:
    out: BufferRef[F]    # forward: normalized output; backward: input gradient
    inp: BufferRef[F]    # [N, C] layer input
    w: BufferRef[F]      # gamma[C], beta[C]
    stats: BufferRef[F]  # [N, 2] mean, rstd
    dout: BufferRef[F]   # backward: output gradient
    part: BufferRef[F]   # backward: [K, 2C] partial sums
    N: uint32
    C: uint32
    R: uint32
    accumulate: uint32   # backward: add into out instead of overwriting


@compute_shader(local_size=(WG, 1, 1))
def layernorm_fwd(global_id: uvec3, p: LNPush):
    n = flat_index(global_id)
    if n >= p.N:
        return
    base = n * p.C
    mean = 0.0
    for c in range(p.C):
        mean += p.inp[base + uint32(c)].v
    mean = mean / float32(p.C)
    var = 0.0
    for c in range(p.C):
        d = p.inp[base + uint32(c)].v - mean
        var += d * d
    rstd = inversesqrt(var / float32(p.C) + 1e-5)
    for c in range(p.C):
        cc = uint32(c)
        xhat = (p.inp[base + cc].v - mean) * rstd
        p.out[base + cc].v = xhat * p.w[cc].v + p.w[p.C + cc].v
    p.stats[n * uint32(2)].v = mean
    p.stats[n * uint32(2) + uint32(1)].v = rstd


@compute_shader(local_size=(WG, 1, 1))
def layernorm_bwd(global_id: uvec3, p: LNPush):
    n = flat_index(global_id)
    if n >= p.N:
        return
    base = n * p.C
    mean = p.stats[n * uint32(2)].v
    rstd = p.stats[n * uint32(2) + uint32(1)].v
    dmean = 0.0
    dxmean = 0.0
    for c in range(p.C):
        cc = uint32(c)
        dn = p.w[cc].v * p.dout[base + cc].v
        dmean += dn
        dxmean += dn * (p.inp[base + cc].v - mean) * rstd
    dmean = dmean / float32(p.C)
    dxmean = dxmean / float32(p.C)
    for c in range(p.C):
        cc = uint32(c)
        xhat = (p.inp[base + cc].v - mean) * rstd
        g = rstd * (p.w[cc].v * p.dout[base + cc].v - dmean - xhat * dxmean)
        if p.accumulate != uint32(0):
            g += p.out[base + cc].v
        p.out[base + cc].v = g


@compute_shader(local_size=(WG, 1, 1))
def layernorm_bwd_partial(global_id: uvec3, p: LNPush):
    j = global_id.x
    k = global_id.y
    if j >= uint32(2) * p.C:
        return
    c = j % p.C
    acc = 0.0
    for r in range(min(p.R, p.N - k * p.R)):
        n = k * p.R + uint32(r)
        d = p.dout[n * p.C + c].v
        if j < p.C:
            d = d * (p.inp[n * p.C + c].v - p.stats[n * uint32(2)].v) \
                * p.stats[n * uint32(2) + uint32(1)].v
        acc += d
    p.part[k * uint32(2) * p.C + j].v = acc


# -- linear layers ------------------------------------------------------------

@gpu_struct
class MMPush:
    out: BufferRef[F]    # fwd: [N, OC]; bwd input grad: [N, C]; bwd weights: partials
    inp: BufferRef[F]    # fwd/bwd weights: [N, C] input; bwd input grad: dout [N, OC]
    w: BufferRef[F]      # W[OC, C], b[OC]
    res: BufferRef[F]    # fwd: residual added to the output
    pre: BufferRef[F]    # fwd: pre-GELU output; bwd: pre-GELU input
    dout: BufferRef[F]   # bwd weights: [N, OC]
    inp4: BufferRef[F4]  # fwd: inp and w as vec4s (C is a multiple of 4)
    w4: BufferRef[F4]
    N: uint32
    C: uint32
    OC: uint32
    flags: uint32        # 1: add residual, 2: GELU
    R: uint32


@compute_shader(local_size=(WG, 1, 1))
def matmul_fwd(global_id: uvec3, p: MMPush):
    i = flat_index(global_id)
    if i >= p.N * p.OC:
        return
    n = i / p.OC
    oc = i % p.OC
    acc = p.w[p.OC * p.C + oc].v
    C4 = p.C / uint32(4)
    for c in range(C4):
        acc += dot(p.inp4[n * C4 + uint32(c)].v, p.w4[oc * C4 + uint32(c)].v)
    if (p.flags & uint32(2)) != uint32(0):
        p.pre[i].v = acc
        acc = gelu(acc)
    if (p.flags & uint32(1)) != uint32(0):
        acc += p.res[i].v
    p.out[i].v = acc


@compute_shader(local_size=(WG, 1, 1))
def matmul_bwd_inp(global_id: uvec3, p: MMPush):
    i = flat_index(global_id)
    if i >= p.N * p.C:
        return
    n = i / p.C
    c = i % p.C
    acc = 0.0
    for oc in range(p.OC):
        acc += p.inp[n * p.OC + uint32(oc)].v * p.w[uint32(oc) * p.C + c].v
    if (p.flags & uint32(2)) != uint32(0):
        acc = acc * gelu_grad(p.pre[i].v)
    p.out[i].v = acc


@compute_shader(local_size=(WG, 1, 1))
def matmul_bwd_w_partial(global_id: uvec3, p: MMPush):
    j = global_id.x
    k = global_id.y
    M = p.OC * p.C + p.OC
    if j >= M:
        return
    acc = 0.0
    rows = min(p.R, p.N - k * p.R)
    if j < p.OC * p.C:
        oc = j / p.C
        c = j % p.C
        for r in range(rows):
            n = k * p.R + uint32(r)
            acc += p.dout[n * p.OC + oc].v * p.inp[n * p.C + c].v
    else:
        oc = j - p.OC * p.C
        for r in range(rows):
            acc += p.dout[(k * p.R + uint32(r)) * p.OC + oc].v
    p.out[k * M + j].v = acc


# -- causal self-attention ----------------------------------------------------
# qkv rows are [q | k | v], each C wide with H heads of hs = C/H. The attention
# matrix of head (b, h) is stored transposed, att[b, h, t2, t] (key t2, query t),
# so invocations for neighbouring queries touch neighbouring addresses.
# ``qkv4``/``dout4`` alias ``qkv``/``dout`` as vec4s (C and hs are multiples of 4).

@gpu_struct
class AttnPush:
    out: BufferRef[F]    # fwd: [N, C]; bwd: dqkv [N, 3C]
    qkv: BufferRef[F]
    att: BufferRef[F]    # softmax probabilities
    datt: BufferRef[F]   # bwd: d(att), then d(pre-softmax scores)
    dout: BufferRef[F]   # bwd: gradient of the attention output [N, C]
    qkv4: BufferRef[F4]
    dout4: BufferRef[F4]
    B: uint32
    T: uint32
    C: uint32
    H: uint32


@compute_shader(local_size=(WG, 1, 1))
def attn_softmax(global_id: uvec3, p: AttnPush):
    row = flat_index(global_id)  # (b*H + h)*T + t
    if row >= p.B * p.H * p.T:
        return
    t = row % p.T
    bh = row / p.T
    h = bh % p.H
    b = bh / p.H
    hs4 = p.C / p.H / uint32(4)
    C34 = uint32(3) * p.C / uint32(4)
    scale = inversesqrt(float32(p.C / p.H))
    q0 = (b * p.T + t) * C34 + h * hs4
    base = bh * p.T * p.T + t
    mx = -1e30
    for t2 in range(t + uint32(1)):
        k0 = (b * p.T + uint32(t2)) * C34 + p.C / uint32(4) + h * hs4
        s = 0.0
        for d in range(hs4):
            s += dot(p.qkv4[q0 + uint32(d)].v, p.qkv4[k0 + uint32(d)].v)
        s = s * scale
        p.att[base + uint32(t2) * p.T].v = s
        mx = max(mx, s)
    total = 0.0
    for t2 in range(t + uint32(1)):
        e = exp(p.att[base + uint32(t2) * p.T].v - mx)
        p.att[base + uint32(t2) * p.T].v = e
        total += e
    inv = 1.0 / total
    for t2 in range(t + uint32(1)):
        p.att[base + uint32(t2) * p.T].v = p.att[base + uint32(t2) * p.T].v * inv


@compute_shader(local_size=(WG, 1, 1))
def attn_out(global_id: uvec3, p: AttnPush):
    i = flat_index(global_id)
    if i >= p.B * p.T * p.C:
        return
    n = i / p.C
    c = i % p.C
    t = n % p.T
    b = n / p.T
    h = c / (p.C / p.H)
    C3 = uint32(3) * p.C
    base = (b * p.H + h) * p.T * p.T + t
    acc = 0.0
    for t2 in range(t + uint32(1)):
        acc += p.att[base + uint32(t2) * p.T].v \
            * p.qkv[(b * p.T + uint32(t2)) * C3 + uint32(2) * p.C + c].v
    p.out[i].v = acc


@compute_shader(local_size=(WG, 1, 1))
def attn_bwd_datt(global_id: uvec3, p: AttnPush):
    i = flat_index(global_id)  # ((b*H + h)*T + t2)*T + t
    if i >= p.B * p.H * p.T * p.T:
        return
    t = i % p.T
    t2 = (i / p.T) % p.T
    if t2 > t:
        return
    bh = i / (p.T * p.T)
    h = bh % p.H
    b = bh / p.H
    hs4 = p.C / p.H / uint32(4)
    o0 = (b * p.T + t) * p.C / uint32(4) + h * hs4
    v0 = ((b * p.T + t2) * uint32(3) * p.C + uint32(2) * p.C) / uint32(4) + h * hs4
    acc = 0.0
    for d in range(hs4):
        acc += dot(p.dout4[o0 + uint32(d)].v, p.qkv4[v0 + uint32(d)].v)
    p.datt[i].v = acc


@compute_shader(local_size=(WG, 1, 1))
def attn_bwd_softmax(global_id: uvec3, p: AttnPush):
    row = flat_index(global_id)  # (b*H + h)*T + t
    if row >= p.B * p.H * p.T:
        return
    t = row % p.T
    base = (row / p.T) * p.T * p.T + t
    scale = inversesqrt(float32(p.C / p.H))
    s = 0.0
    for t2 in range(t + uint32(1)):
        j = base + uint32(t2) * p.T
        s += p.att[j].v * p.datt[j].v
    for t2 in range(t + uint32(1)):
        j = base + uint32(t2) * p.T
        p.datt[j].v = p.att[j].v * (p.datt[j].v - s) * scale


@compute_shader(local_size=(WG, 1, 1))
def attn_bwd_qkv(global_id: uvec3, p: AttnPush):
    C3 = uint32(3) * p.C
    i = flat_index(global_id)
    if i >= p.B * p.T * C3:
        return
    n = i / C3
    j = i % C3
    part = j / p.C
    c = j % p.C
    t = n % p.T
    b = n / p.T
    h = c / (p.C / p.H)
    bh = (b * p.H + h) * p.T   # (query t, key t2) lives at (bh + t2) * T + t
    acc = 0.0
    if part == uint32(0):    # dq[t] = sum_{t2<=t} dS[t, t2] k[t2]
        for t2 in range(t + uint32(1)):
            acc += p.datt[(bh + uint32(t2)) * p.T + t].v \
                * p.qkv[(b * p.T + uint32(t2)) * C3 + p.C + c].v
    elif part == uint32(1):  # dk[t] = sum_{t1>=t} dS[t1, t] q[t1]
        for t1 in range(t, p.T):
            acc += p.datt[(bh + t) * p.T + uint32(t1)].v \
                * p.qkv[(b * p.T + uint32(t1)) * C3 + c].v
    else:                    # dv[t] = sum_{t1>=t} att[t1, t] dout[t1]
        for t1 in range(t, p.T):
            acc += p.att[(bh + t) * p.T + uint32(t1)].v \
                * p.dout[(b * p.T + uint32(t1)) * p.C + c].v
    p.out[i].v = acc


# -- loss and optimizer -------------------------------------------------------

@gpu_struct
class CEPush:
    logits: BufferRef[F]   # [N, V]; replaced by d(loss)/d(logits) when grad != 0
    losses: BufferRef[F]   # [N]
    targets: BufferRef[U]  # [N] packed bytes
    N: uint32
    V: uint32
    grad: uint32
    scale: float32         # d(loss)/d(per-token loss); 1/(N * accumulation steps)


@compute_shader(local_size=(WG, 1, 1))
def cross_entropy(global_id: uvec3, p: CEPush):
    n = flat_index(global_id)
    if n >= p.N:
        return
    base = n * p.V
    target = byte_at(p.targets, n)
    mx = -1e30
    for v in range(p.V):
        mx = max(mx, p.logits[base + uint32(v)].v)
    total = 0.0
    for v in range(p.V):
        total += exp(p.logits[base + uint32(v)].v - mx)
    p.losses[n].v = log(total) + mx - p.logits[base + target].v
    if p.grad != uint32(0):
        scale = p.scale
        inv = 1.0 / total
        for v in range(p.V):
            prob = exp(p.logits[base + uint32(v)].v - mx) * inv
            if uint32(v) == target:
                prob -= 1.0
            p.logits[base + uint32(v)].v = prob * scale


@gpu_struct
class NormPush:
    out: BufferRef[F]    # [stride] partial sums of squares
    grads: BufferRef[F]
    n: uint32
    stride: uint32


@compute_shader(local_size=(WG, 1, 1))
def sumsq_partial(global_id: uvec3, p: NormPush):
    j = flat_index(global_id)
    if j >= p.stride:
        return
    acc = 0.0
    i = j
    while i < p.n:
        g = p.grads[i].v
        acc += g * g
        i += p.stride
    p.out[j].v = acc


@gpu_struct
class AdamPush:
    params: BufferRef[F]
    grads: BufferRef[F]
    m: BufferRef[F]
    v: BufferRef[F]
    hyper: BufferRef[Hyper]
    decay: BufferRef[F]    # 1 for weight matrices and embeddings, 0 for biases and norms
    normsq: BufferRef[F]   # [1] squared global gradient norm
    n: uint32


@compute_shader(local_size=(WG, 1, 1))
def adam(global_id: uvec3, p: AdamPush):
    i = flat_index(global_id)
    if i >= p.n:
        return
    hp = p.hyper[uint32(0)]
    g = p.grads[i].v
    if hp.clip > 0.0:
        g = g * min(1.0, hp.clip / (sqrt(p.normsq[uint32(0)].v) + 1e-6))
    m = hp.beta1 * p.m[i].v + (1.0 - hp.beta1) * g
    v = hp.beta2 * p.v[i].v + (1.0 - hp.beta2) * g * g
    p.m[i].v = m
    p.v[i].v = v
    w = p.params[i].v
    p.params[i].v = w - hp.lr * ((m / hp.bc1) / (sqrt(v / hp.bc2) + hp.eps)
                                 + hp.wd * p.decay[i].v * w)
