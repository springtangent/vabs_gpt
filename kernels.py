"""Compute shaders for a GPT-style transformer: forward, backward and Adam.

All tensors are flat float32 arrays addressed through ``BufferRef`` push
constants, row-major, with ``N = B*T`` token rows. Buffers are read and written
as ``vec4`` (``F4``) wherever a kernel walks contiguous channels; every channel
count (C, head size, MLP width, vocab) is a multiple of 4.

The shader language has no workgroup memory, so the matrix products are
register-tiled instead: one invocation computes a 4x4 block of outputs (4 rows
x 4 channels), loading four ``vec4`` from each operand per step and combining
them with one ``mat4`` product per row. That is 64 multiply-adds for 8 loads,
where one invocation per output element needs 2 loads per 4. Row indices past
the end are clamped for loads and skipped for stores, so row counts need not
be multiples of 4.

Weight gradients sum over the N rows in two passes: a ``*_partial`` kernel
writes one partial sum per chunk of R rows (dispatch y = chunk), and
``reduce_partials`` adds the chunks. Chunked kernels allow a short last chunk.
Weight gradients accumulate (``+=``) so micro-batches can be summed; the
trainer zeroes them before each optimizer step.

A linear layer's parameters are ``W[OC, C]`` followed by ``b[OC]``; a layer
norm's are ``gamma[C]`` followed by ``beta[C]``. Partial sums follow the same
order, so one reduction writes the weight and bias grads.

Element-wise kernels index with ``flat_index``: a dispatch wider than
``SPLIT_GROUPS`` workgroups continues in global_id.y (``Device.run`` splits it),
which keeps large batches and contexts under the 65535-group dispatch limit.

Tokens are bytes packed four per ``uint32`` word (little endian).
Shaders cannot read Python globals, so all sizes come in push constants.
Integer ``/`` in a shader is integer division (the DSL has no ``//``).
"""
from vabs.shaders import (compute_shader, shader_fn, dot, exp, inversesqrt, log, max, min, mix,
                          sqrt, step)
from vabs.structs import BufferRef, float32, gpu_struct, mat4, uint32, uvec3, vec4

WG = 256      # local size of the per-element and per-tile kernels
ROW_WG = 64   # local size of kernels that loop over a whole row: there are few rows, and
              # smaller groups spread them over more of the GPU
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
def flat_row(gid: uvec3) -> uint32:
    return gid.x + gid.y * uint32(2097152)  # ROW_WG * SPLIT_GROUPS


@shader_fn
def byte_at(buf: BufferRef[U], i: uint32) -> uint32:
    return (buf[i >> uint32(2)].v >> ((i & uint32(3)) * uint32(8))) & uint32(255)


@shader_fn
def tanh_(x: vec4) -> vec4:
    return 1.0 - 2.0 / (exp(2.0 * x) + 1.0)


@shader_fn
def gelu(x: vec4) -> vec4:
    return 0.5 * x * (1.0 + tanh_(0.7978845608 * (x + 0.044715 * x * x * x)))


@shader_fn
def gelu_grad(x: vec4) -> vec4:
    th = tanh_(0.7978845608 * (x + 0.044715 * x * x * x))
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * 0.7978845608 * (
        1.0 + 3.0 * 0.044715 * x * x)


@shader_fn
def hsum(x: vec4) -> float32:
    return dot(x, vec4(1.0))


@shader_fn
def hmax(x: vec4) -> float32:
    return max(max(x.x, x.y), max(x.z, x.w))


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


@gpu_struct
class Reduce4Push:
    out: BufferRef[F4]
    part: BufferRef[F4]
    M4: uint32  # vec4s per chunk
    K: uint32   # chunks


@compute_shader(local_size=(WG, 1, 1))
def reduce_partials4(global_id: uvec3, p: Reduce4Push):
    """out[j] += sum over chunks of part[k, j], four values at a time."""
    j = flat_index(global_id)
    if j >= p.M4:
        return
    acc = p.out[j].v
    for k in range(p.K):
        acc += p.part[uint32(k) * p.M4 + j].v
    p.out[j].v = acc


@gpu_struct
class GatherPush:
    out: BufferRef[F]
    src: BufferRef[F]
    idx: BufferRef[U]
    n: uint32


@compute_shader(local_size=(WG, 1, 1))
def gather(global_id: uvec3, p: GatherPush):
    """out[i] = src[idx[i]]; builds the transposed weight copies after each update."""
    i = flat_index(global_id)
    if i >= p.n:
        return
    p.out[i].v = p.src[p.idx[i].v].v


# -- embeddings ---------------------------------------------------------------

@gpu_struct
class EmbedPush:
    x: BufferRef[F]      # [N, C] output (forward) or its gradient (backward)
    wte: BufferRef[F]    # [V, C] token table
    wpe: BufferRef[F]    # [T, C] position table or its gradient
    toks: BufferRef[U]   # [N] packed bytes
    x4: BufferRef[F4]    # backward: x as vec4s
    part: BufferRef[F4]  # backward: partial sums [K, V*C]
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
    j = global_id.x   # (v, c/4)
    k = global_id.y
    M4 = p.M / uint32(4)
    if j >= M4:
        return
    C4 = p.C / uint32(4)
    v = j / C4
    c = j % C4
    acc = vec4(0.0)
    for r in range(min(p.R, p.N - k * p.R)):
        n = k * p.R + uint32(r)
        if byte_at(p.toks, n) == v:
            acc += p.x4[n * C4 + c].v
    p.part[k * M4 + j].v = acc


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
    out: BufferRef[F4]   # forward: normalized output; backward: input gradient
    inp: BufferRef[F4]   # [N, C] layer input
    w: BufferRef[F4]     # gamma[C], beta[C]
    stats: BufferRef[F]  # [N, 2] mean, rstd
    dout: BufferRef[F4]  # backward: output gradient
    part: BufferRef[F4]  # backward: [K, 2C] partial sums
    N: uint32
    C: uint32
    R: uint32
    accumulate: uint32   # backward: add into out instead of overwriting


@compute_shader(local_size=(ROW_WG, 1, 1))
def layernorm_fwd(global_id: uvec3, p: LNPush):
    n = flat_row(global_id)
    if n >= p.N:
        return
    C4 = p.C / uint32(4)
    base = n * C4
    s = vec4(0.0)
    for c in range(C4):
        s += p.inp[base + uint32(c)].v
    mean = hsum(s) / float32(p.C)
    s = vec4(0.0)
    for c in range(C4):
        d = p.inp[base + uint32(c)].v - mean
        s += d * d
    rstd = inversesqrt(hsum(s) / float32(p.C) + 1e-5)
    for c in range(C4):
        cc = uint32(c)
        p.out[base + cc].v = (p.inp[base + cc].v - mean) * rstd * p.w[cc].v + p.w[C4 + cc].v
    p.stats[n * uint32(2)].v = mean
    p.stats[n * uint32(2) + uint32(1)].v = rstd


@compute_shader(local_size=(ROW_WG, 1, 1))
def layernorm_bwd(global_id: uvec3, p: LNPush):
    n = flat_row(global_id)
    if n >= p.N:
        return
    C4 = p.C / uint32(4)
    base = n * C4
    mean = p.stats[n * uint32(2)].v
    rstd = p.stats[n * uint32(2) + uint32(1)].v
    sm = vec4(0.0)
    sx = vec4(0.0)
    for c in range(C4):
        cc = uint32(c)
        dn = p.w[cc].v * p.dout[base + cc].v
        sm += dn
        sx += dn * (p.inp[base + cc].v - mean)
    dmean = hsum(sm) / float32(p.C)
    dxmean = hsum(sx) * rstd / float32(p.C)
    for c in range(C4):
        cc = uint32(c)
        xhat = (p.inp[base + cc].v - mean) * rstd
        g = rstd * (p.w[cc].v * p.dout[base + cc].v - dmean - xhat * dxmean)
        if p.accumulate != uint32(0):
            g += p.out[base + cc].v
        p.out[base + cc].v = g


@compute_shader(local_size=(ROW_WG, 1, 1))
def layernorm_bwd_partial(global_id: uvec3, p: LNPush):
    c = global_id.x   # c/4: writes 4 gamma and 4 beta partial sums
    k = global_id.y
    C4 = p.C / uint32(4)
    if c >= C4:
        return
    gg = vec4(0.0)
    gb = vec4(0.0)
    for r in range(min(p.R, p.N - k * p.R)):
        n = k * p.R + uint32(r)
        d = p.dout[n * C4 + c].v
        gb += d
        gg += d * (p.inp[n * C4 + c].v - p.stats[n * uint32(2)].v) \
            * p.stats[n * uint32(2) + uint32(1)].v
    p.part[k * uint32(2) * C4 + c].v = gg
    p.part[k * uint32(2) * C4 + C4 + c].v = gb


# -- linear layers ------------------------------------------------------------

@gpu_struct
class MMPush:
    out: BufferRef[F4]   # fwd: [N, OC]; bwd input grad: [N, C]; bwd weights: partials
    inp: BufferRef[F4]   # fwd/bwd weights: [N, C] input; bwd input grad: dout [N, OC]
    w: BufferRef[F4]     # W[OC, C], b[OC]; fwd: the transposed copy W^T[C, OC], b[OC]
    res: BufferRef[F4]   # fwd: residual added to the output
    pre: BufferRef[F4]   # fwd: pre-GELU output; bwd: pre-GELU input
    dout: BufferRef[F4]  # bwd weights: [N, OC]
    N: uint32
    C: uint32
    OC: uint32
    flags: uint32        # 1: add residual, 2: GELU
    R: uint32


@shader_fn(inline=True)
def store_act(out: BufferRef[F4], pre: BufferRef[F4], res: BufferRef[F4], i: uint32, a: vec4,
              flags: uint32):
    """out[i] = (GELU if flags & 2)(a) (+ res[i] if flags & 1); pre[i] keeps the GELU input."""
    x = a
    if (flags & uint32(2)) != uint32(0):
        pre[i].v = x
        x = gelu(x)
    if (flags & uint32(1)) != uint32(0):
        x += res[i].v
    out[i].v = x


@compute_shader(local_size=(WG, 1, 1))
def matmul_fwd(global_id: uvec3, p: MMPush):
    """out[n, oc] = act(inp[n, :] . W[oc, :] + b[oc]) (+ res) for 4 rows x 4 outputs.

    Reads the transposed weights: at each step the invocations of a workgroup
    load the same four inputs per row and neighbouring vec4s of four W^T rows.
    (4 x 8 tiles were measured slower, and so were explicit multiply-add chains in place of
    the mat4 product, which serialize on the accumulator.)
    """
    i = flat_index(global_id)
    C4 = p.C / uint32(4)
    OC4 = p.OC / uint32(4)
    n0 = (i / OC4) * uint32(4)
    if n0 >= p.N:
        return
    o = i % OC4
    last = p.N - uint32(1)
    n1 = min(n0 + uint32(1), last)
    n2 = min(n0 + uint32(2), last)
    n3 = min(n0 + uint32(3), last)
    r0 = n0 * C4
    r1 = n1 * C4
    r2 = n2 * C4
    r3 = n3 * C4
    a0 = p.w[OC4 * p.C + o].v
    a1 = a0
    a2 = a0
    a3 = a0
    for c in range(C4):
        cc = uint32(c)
        w0 = cc * uint32(4) * OC4 + o
        W = mat4(p.w[w0].v, p.w[w0 + OC4].v, p.w[w0 + uint32(2) * OC4].v,
                 p.w[w0 + uint32(3) * OC4].v)
        a0 += W * p.inp[r0 + cc].v
        a1 += W * p.inp[r1 + cc].v
        a2 += W * p.inp[r2 + cc].v
        a3 += W * p.inp[r3 + cc].v
    store_act(p.out, p.pre, p.res, n0 * OC4 + o, a0, p.flags)
    if n0 + uint32(1) <= last:
        store_act(p.out, p.pre, p.res, n1 * OC4 + o, a1, p.flags)
    if n0 + uint32(2) <= last:
        store_act(p.out, p.pre, p.res, n2 * OC4 + o, a2, p.flags)
    if n0 + uint32(3) <= last:
        store_act(p.out, p.pre, p.res, n3 * OC4 + o, a3, p.flags)


@compute_shader(local_size=(WG, 1, 1))
def matmul_bwd_inp(global_id: uvec3, p: MMPush):
    """out[n, c] = sum_oc inp[n, oc] W[oc, c] (* gelu'(pre)) for 4 rows x 4 channels."""
    i = flat_index(global_id)
    C4 = p.C / uint32(4)
    OC4 = p.OC / uint32(4)
    n0 = (i / C4) * uint32(4)
    if n0 >= p.N:
        return
    c = i % C4
    last = p.N - uint32(1)
    n1 = min(n0 + uint32(1), last)
    n2 = min(n0 + uint32(2), last)
    n3 = min(n0 + uint32(3), last)
    r0 = n0 * OC4
    r1 = n1 * OC4
    r2 = n2 * OC4
    r3 = n3 * OC4
    a0 = vec4(0.0)
    a1 = vec4(0.0)
    a2 = vec4(0.0)
    a3 = vec4(0.0)
    for o in range(OC4):
        oo = uint32(o)
        w0 = oo * uint32(4) * C4 + c
        W = mat4(p.w[w0].v, p.w[w0 + C4].v, p.w[w0 + uint32(2) * C4].v,
                 p.w[w0 + uint32(3) * C4].v)
        a0 += W * p.inp[r0 + oo].v
        a1 += W * p.inp[r1 + oo].v
        a2 += W * p.inp[r2 + oo].v
        a3 += W * p.inp[r3 + oo].v
    if (p.flags & uint32(2)) != uint32(0):
        a0 = a0 * gelu_grad(p.pre[n0 * C4 + c].v)
        a1 = a1 * gelu_grad(p.pre[n1 * C4 + c].v)
        a2 = a2 * gelu_grad(p.pre[n2 * C4 + c].v)
        a3 = a3 * gelu_grad(p.pre[n3 * C4 + c].v)
    p.out[n0 * C4 + c].v = a0
    if n0 + uint32(1) <= last:
        p.out[n1 * C4 + c].v = a1
    if n0 + uint32(2) <= last:
        p.out[n2 * C4 + c].v = a2
    if n0 + uint32(3) <= last:
        p.out[n3 * C4 + c].v = a3


@compute_shader(local_size=(WG, 1, 1))
def matmul_bwd_w_partial(global_id: uvec3, p: MMPush):
    """Partial sums over one chunk of rows of dW[oc, c] = dout[n, oc] inp[n, c] and db.

    x indexes 4x4 tiles of dW (OC/4 * C/4 of them), then the OC/4 vec4s of db.
    """
    j = global_id.x
    k = global_id.y
    C4 = p.C / uint32(4)
    OC4 = p.OC / uint32(4)
    tiles = OC4 * C4
    if j >= tiles + OC4:
        return
    M4 = OC4 * p.C + OC4
    n0 = k * p.R
    rows = min(p.R, p.N - n0)
    if j < tiles:
        o = j / C4
        c = j % C4
        d0 = n0 * OC4 + o
        x0 = n0 * C4 + c
        a0 = vec4(0.0)
        a1 = vec4(0.0)
        a2 = vec4(0.0)
        a3 = vec4(0.0)
        for r in range(rows):
            rr = uint32(r)
            d = p.dout[d0 + rr * OC4].v
            x = p.inp[x0 + rr * C4].v
            a0 += d.x * x
            a1 += d.y * x
            a2 += d.z * x
            a3 += d.w * x
        base = k * M4 + o * uint32(4) * C4 + c
        p.out[base].v = a0
        p.out[base + C4].v = a1
        p.out[base + uint32(2) * C4].v = a2
        p.out[base + uint32(3) * C4].v = a3
    else:
        o = j - tiles
        acc = vec4(0.0)
        for r in range(rows):
            acc += p.dout[(n0 + uint32(r)) * OC4 + o].v
        p.out[k * M4 + OC4 * p.C + o].v = acc


# -- causal self-attention ----------------------------------------------------
# qkv rows are [q | k | v], each C wide with H heads of hs = C/H. The attention
# matrix of head (b, h) is att[b, h, query t, key t2] with rows padded to a
# multiple of 4 keys. Entries after the diagonal up to the end of their vec4
# are written as zeros, so tiled kernels can use whole vec4s of keys.

@gpu_struct
class AttnPush:
    qkv: BufferRef[F4]
    att: BufferRef[F4]   # softmax probabilities
    datt: BufferRef[F4]  # bwd: d(pre-softmax scores)
    dout: BufferRef[F4]  # bwd: gradient of the attention output [N, C]
    y: BufferRef[F4]     # the attention output [N, C] (written by attn_fwd)
    smax: BufferRef[F]   # fwd: per query row, the largest scaled score (for QK-Clip)
    B: uint32
    T: uint32
    C: uint32
    H: uint32
    keep: uint32         # attn_fwd: store the attention matrix
    keepmax: uint32      # fwd: store each row's largest score in smax


@shader_fn(inline=True)
def key_dots(a: BufferRef[F4], a0: uint32, rows: BufferRef[F4], r0: uint32, stride: uint32,
             t2: uint32, last: uint32, hs4: uint32) -> vec4:
    """a[a0 : a0 + hs4] . rows[r0 + t * stride : ... + hs4] for t = t2 .. t2 + 3 (clamped)."""
    k0 = r0 + t2 * stride
    k1 = r0 + min(t2 + uint32(1), last) * stride
    k2 = r0 + min(t2 + uint32(2), last) * stride
    k3 = r0 + min(t2 + uint32(3), last) * stride
    s = vec4(0.0)
    for d in range(hs4):
        dd = uint32(d)
        s += a[a0 + dd].v * mat4(rows[k0 + dd].v, rows[k1 + dd].v, rows[k2 + dd].v,
                                 rows[k3 + dd].v)
    return s


@compute_shader(local_size=(WG, 1, 1))
def attn_softmax(global_id: uvec3, p: AttnPush):
    """att[t, :] = softmax over keys t2 <= t of q[t] . k[t2] / sqrt(hs), one query row."""
    row = flat_index(global_id)  # (b*H + h)*T + t
    if row >= p.B * p.H * p.T:
        return
    t = row % p.T
    bh = row / p.T
    h = bh % p.H
    b = bh / p.H
    C4 = p.C / uint32(4)
    hs4 = C4 / p.H
    C34 = uint32(3) * C4
    scale = inversesqrt(float32(p.C / p.H))
    last = p.T - uint32(1)
    q0 = (b * p.T + t) * C34 + h * hs4
    kb0 = b * p.T * C34 + C4 + h * hs4
    a0 = row * ((p.T + uint32(3)) / uint32(4))
    blocks = t / uint32(4) + uint32(1)
    ft = vec4(float32(t))
    # The scores are computed twice (running max and sum, then the probabilities)
    # so that the attention row is written once and never read back here.
    mx = -1e30
    total = 0.0
    for kb in range(blocks):
        t2 = uint32(kb) * uint32(4)
        s = key_dots(p.qkv, q0, p.qkv, kb0, C34, t2, last, hs4) * scale
        idx = vec4(float32(t2)) + vec4(0.0, 1.0, 2.0, 3.0)
        s = mix(vec4(-1e30), s, step(idx, ft))  # keys after t: exp() gives 0
        m = max(mx, hmax(s))
        total = total * exp(mx - m) + hsum(exp(s - m))
        mx = m
    if p.keepmax != uint32(0):
        p.smax[row].v = mx
    inv = 1.0 / total
    for kb in range(blocks):
        t2 = uint32(kb) * uint32(4)
        s = key_dots(p.qkv, q0, p.qkv, kb0, C34, t2, last, hs4) * scale
        idx = vec4(float32(t2)) + vec4(0.0, 1.0, 2.0, 3.0)
        s = mix(vec4(-1e30), s, step(idx, ft))
        p.att[a0 + uint32(kb)].v = exp(s - mx) * inv


@compute_shader(local_size=(WG, 1, 1))
def attn_bwd_scores(global_id: uvec3, p: AttnPush):
    """datt[t, :] = d(loss)/d(scores) of one query row.

    d(att)[t, t2] = dout[t] . v[t2], pushed back through the softmax and scale:
    datt = att * (d(att) - sum_t2 att d(att)) * scale. That sum equals dout[t] . y[t],
    because y[t] = sum_t2 att[t, t2] v[t2].
    """
    row = flat_index(global_id)  # (b*H + h)*T + t
    if row >= p.B * p.H * p.T:
        return
    t = row % p.T
    bh = row / p.T
    h = bh % p.H
    b = bh / p.H
    C4 = p.C / uint32(4)
    hs4 = C4 / p.H
    C34 = uint32(3) * C4
    scale = inversesqrt(float32(p.C / p.H))
    last = p.T - uint32(1)
    o0 = (b * p.T + t) * C4 + h * hs4
    vb0 = b * p.T * C34 + uint32(2) * C4 + h * hs4
    a0 = row * ((p.T + uint32(3)) / uint32(4))
    blocks = t / uint32(4) + uint32(1)
    total = 0.0
    for d in range(hs4):
        total += dot(p.dout[o0 + uint32(d)].v, p.y[o0 + uint32(d)].v)
    for kb in range(blocks):   # att is 0 after the diagonal, so datt is too
        kk = uint32(kb)
        s = key_dots(p.dout, o0, p.qkv, vb0, C34, kk * uint32(4), last, hs4)
        p.datt[a0 + kk].v = p.att[a0 + kk].v * (s - total) * scale


@shader_fn(inline=True)
def rows_times(rows: BufferRef[F4], r0: uint32, stride: uint32, t2: uint32, last: uint32,
               w: vec4) -> vec4:
    """sum_i w[i] * rows[r0 + (t2 + i) * stride] for i = 0 .. 3 (row index clamped to last)."""
    return mat4(rows[r0 + t2 * stride].v,
                rows[r0 + min(t2 + uint32(1), last) * stride].v,
                rows[r0 + min(t2 + uint32(2), last) * stride].v,
                rows[r0 + min(t2 + uint32(3), last) * stride].v) * w


@compute_shader(local_size=(WG, 1, 1))
def attn_fwd(global_id: uvec3, p: AttnPush):
    """``attn_softmax`` and the attention output y[t] = sum_t2 att[t, t2] v[t2] in one pass.

    For head sizes up to 16: the output of one head is at most four vec4s, which
    the invocation of a query row keeps in registers. The attention row is
    written only when ``keep`` is set (the backward pass reads it), which makes
    evaluation and sampling about 20% faster than the two separate kernels. In
    training it is about 13% faster than them. The same fusion of
    ``attn_bwd_scores`` with the query gradient measured slower than the
    separate kernels.
    """
    row = flat_index(global_id)  # (b*H + h)*T + t
    if row >= p.B * p.H * p.T:
        return
    t = row % p.T
    bh = row / p.T
    h = bh % p.H
    b = bh / p.H
    C4 = p.C / uint32(4)
    hs4 = C4 / p.H
    C34 = uint32(3) * C4
    scale = inversesqrt(float32(p.C / p.H))
    last = p.T - uint32(1)
    q0 = (b * p.T + t) * C34 + h * hs4
    kb0 = b * p.T * C34 + C4 + h * hs4
    vb0 = kb0 + C4
    a0 = row * ((p.T + uint32(3)) / uint32(4))
    blocks = t / uint32(4) + uint32(1)
    ft = vec4(float32(t))
    mx = -1e30
    total = 0.0
    for kb in range(blocks):
        t2 = uint32(kb) * uint32(4)
        s = key_dots(p.qkv, q0, p.qkv, kb0, C34, t2, last, hs4) * scale
        idx = vec4(float32(t2)) + vec4(0.0, 1.0, 2.0, 3.0)
        s = mix(vec4(-1e30), s, step(idx, ft))  # keys after t: exp() gives 0
        m = max(mx, hmax(s))
        total = total * exp(mx - m) + hsum(exp(s - m))
        mx = m
    inv = 1.0 / total
    y0 = vec4(0.0)
    y1 = vec4(0.0)
    y2 = vec4(0.0)
    y3 = vec4(0.0)
    for kb in range(blocks):
        t2 = uint32(kb) * uint32(4)
        s = key_dots(p.qkv, q0, p.qkv, kb0, C34, t2, last, hs4) * scale
        idx = vec4(float32(t2)) + vec4(0.0, 1.0, 2.0, 3.0)
        s = mix(vec4(-1e30), s, step(idx, ft))
        w = exp(s - mx) * inv
        if p.keep != uint32(0):
            p.att[a0 + uint32(kb)].v = w
        y0 += rows_times(p.qkv, vb0, C34, t2, last, w)
        if hs4 > uint32(1):
            y1 += rows_times(p.qkv, vb0 + uint32(1), C34, t2, last, w)
        if hs4 > uint32(2):
            y2 += rows_times(p.qkv, vb0 + uint32(2), C34, t2, last, w)
        if hs4 > uint32(3):
            y3 += rows_times(p.qkv, vb0 + uint32(3), C34, t2, last, w)
    o0 = (b * p.T + t) * C4 + h * hs4
    p.y[o0].v = y0
    if hs4 > uint32(1):
        p.y[o0 + uint32(1)].v = y1
    if hs4 > uint32(2):
        p.y[o0 + uint32(2)].v = y2
    if hs4 > uint32(3):
        p.y[o0 + uint32(3)].v = y3
    if p.keepmax != uint32(0):   # stored last: storing it between the loops was slower
        p.smax[row].v = mx


@gpu_struct
class AttnMMPush:
    """Products with an attention-shaped matrix s[b, h, t, t2] (lower triangular)."""
    out: BufferRef[F4]   # first output channel of head 0, row 0
    x: BufferRef[F4]     # first input channel of head 0, row 0
    s: BufferRef[F4]
    B: uint32
    T: uint32
    H: uint32
    hs4: uint32          # head size / 4
    xs: uint32           # vec4s per row of x
    os: uint32           # vec4s per row of out


@compute_shader(local_size=(WG, 1, 1))
def attn_mm(global_id: uvec3, p: AttnMMPush):
    """out[t, c] = sum_{t2 <= t} s[t, t2] x[t2, c] for 4 queries x 4 channels."""
    i = flat_index(global_id)   # ((b*H + h)*TB + t/4)*hs4 + c/4
    TB = (p.T + uint32(3)) / uint32(4)
    if i >= p.B * p.H * TB * p.hs4:
        return
    c = i % p.hs4
    j = i / p.hs4
    tb = j % TB
    bh = j / TB
    hc = (bh % p.H) * p.hs4 + c
    bT = (bh / p.H) * p.T
    last = p.T - uint32(1)
    t0 = tb * uint32(4)
    t1 = min(t0 + uint32(1), last)
    t2 = min(t0 + uint32(2), last)
    t3 = min(t0 + uint32(3), last)
    s0 = (bh * p.T + t0) * TB
    s1 = (bh * p.T + t1) * TB
    s2 = (bh * p.T + t2) * TB
    s3 = (bh * p.T + t3) * TB
    xb = bT * p.xs + hc
    a0 = vec4(0.0)
    a1 = vec4(0.0)
    a2 = vec4(0.0)
    a3 = vec4(0.0)
    for kb in range(tb + uint32(1)):
        kk = uint32(kb)
        k = kk * uint32(4)
        X = mat4(p.x[xb + k * p.xs].v, p.x[xb + min(k + uint32(1), last) * p.xs].v,
                 p.x[xb + min(k + uint32(2), last) * p.xs].v,
                 p.x[xb + min(k + uint32(3), last) * p.xs].v)
        a0 += X * p.s[s0 + kk].v
        a1 += X * p.s[s1 + kk].v
        a2 += X * p.s[s2 + kk].v
        a3 += X * p.s[s3 + kk].v
    ob = bT * p.os + hc
    p.out[ob + t0 * p.os].v = a0
    if t0 + uint32(1) <= last:
        p.out[ob + t1 * p.os].v = a1
    if t0 + uint32(2) <= last:
        p.out[ob + t2 * p.os].v = a2
    if t0 + uint32(3) <= last:
        p.out[ob + t3 * p.os].v = a3


@compute_shader(local_size=(WG, 1, 1))
def attn_mm_t(global_id: uvec3, p: AttnMMPush):
    """out[t2, c] = sum_{t >= t2} s[t, t2] x[t, c] for 4 keys x 4 channels."""
    i = flat_index(global_id)   # ((b*H + h)*TB + t2/4)*hs4 + c/4
    TB = (p.T + uint32(3)) / uint32(4)
    if i >= p.B * p.H * TB * p.hs4:
        return
    c = i % p.hs4
    j = i / p.hs4
    kb = j % TB
    bh = j / TB
    hc = (bh % p.H) * p.hs4 + c
    bT = (bh / p.H) * p.T
    k0 = kb * uint32(4)
    sb = bh * p.T * TB + kb
    xb = bT * p.xs + hc
    a0 = vec4(0.0)
    a1 = vec4(0.0)
    a2 = vec4(0.0)
    a3 = vec4(0.0)
    for r in range(p.T - k0):   # from the last query down: every invocation is on the same row
        tt = p.T - uint32(1) - uint32(r)
        s = p.s[sb + tt * TB].v
        x = p.x[xb + tt * p.xs].v
        a0 += s.x * x
        a1 += s.y * x
        a2 += s.z * x
        a3 += s.w * x
    ob = (bT + k0) * p.os + hc
    p.out[ob].v = a0
    if k0 + uint32(1) < p.T:
        p.out[ob + p.os].v = a1
    if k0 + uint32(2) < p.T:
        p.out[ob + uint32(2) * p.os].v = a2
    if k0 + uint32(3) < p.T:
        p.out[ob + uint32(3) * p.os].v = a3


# -- loss and optimizer -------------------------------------------------------

@gpu_struct
class CEPush:
    logits: BufferRef[F]   # [N, V]; replaced by d(loss)/d(logits) when grad != 0
    logits4: BufferRef[F4]
    losses: BufferRef[F]   # [N]
    targets: BufferRef[U]  # [N] packed bytes
    N: uint32
    V: uint32
    grad: uint32
    scale: float32         # d(loss)/d(per-token loss); 1/(N * accumulation steps)


@compute_shader(local_size=(ROW_WG, 1, 1))
def cross_entropy(global_id: uvec3, p: CEPush):
    n = flat_row(global_id)
    if n >= p.N:
        return
    V4 = p.V / uint32(4)
    base = n * V4
    target = n * p.V + byte_at(p.targets, n)
    mx = -1e30
    for v in range(V4):
        mx = max(mx, hmax(p.logits4[base + uint32(v)].v))
    total = 0.0
    for v in range(V4):
        total += hsum(exp(p.logits4[base + uint32(v)].v - mx))
    p.losses[n].v = log(total) + mx - p.logits[target].v
    if p.grad != uint32(0):
        f = p.scale / total
        for v in range(V4):
            p.logits4[base + uint32(v)].v = exp(p.logits4[base + uint32(v)].v - mx) * f
        p.logits[target].v = p.logits[target].v - p.scale


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
    mask: BufferRef[F]     # with masked != 0: 1 where Adam applies, 0 for Muon's matrices
    n: uint32
    masked: uint32


@compute_shader(local_size=(WG, 1, 1))
def adam(global_id: uvec3, p: AdamPush):
    i = flat_index(global_id)
    if i >= p.n:
        return
    if p.masked != uint32(0):
        if p.mask[i].v == 0.0:
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


# -- Muon ---------------------------------------------------------------------
# Muon (Jordan et al. 2024; "Muon is Scalable for LLM Training", Liu et al. 2025)
# updates each hidden weight matrix with its orthogonalized Nesterov momentum.
# The matrices are copied into a workspace through an index table, transposed
# where needed so that every one is m x n with m <= n, and matrices of equal
# shape form one batch. Newton-Schulz then runs on a whole batch at once:
#     X = M / |M|_F;  5 times: A = X X^T,  B = b A + c A A,  X = a X + B X
# The first iteration takes the norm from trace(X X^T) and scales on the fly. X^T is
# kept beside X so that X X^T is an ordinary product with coalesced loads.
# Every m and n is the model width or the MLP width, both multiples of 4.

@gpu_struct
class MuonPush:
    grads: BufferRef[F]
    m: BufferRef[F]        # the momentum, kept in the first Adam moment's place
    x: BufferRef[F]        # workspace [nb, m, n]
    xt: BufferRef[F]       # its transpose [nb, n, m]
    idx: BufferRef[U]      # workspace element -> parameter index
    hyper: BufferRef[Hyper]
    normsq: BufferRef[F]   # [1] squared global gradient norm (for clipping)
    mu: float32            # momentum
    total: uint32          # workspace elements
    rows: uint32           # m
    cols: uint32           # n


@compute_shader(local_size=(WG, 1, 1))
def muon_momentum(global_id: uvec3, p: MuonPush):
    """m = mu m + g, then x = g + mu m (Nesterov), gathered into the workspace (and X^T)."""
    j = flat_index(global_id)
    if j >= p.total:
        return
    hp = p.hyper[uint32(0)]
    f = 1.0
    if hp.clip > 0.0:
        f = min(1.0, hp.clip / (sqrt(p.normsq[uint32(0)].v) + 1e-6))
    k = p.idx[j].v
    g = p.grads[k].v * f
    mo = p.mu * p.m[k].v + g
    p.m[k].v = mo
    x = g + p.mu * mo
    p.x[j].v = x
    e = j % (p.rows * p.cols)
    p.xt[j - e + (e % p.cols) * p.rows + e / p.cols].v = x


@gpu_struct
class MuonNormPush:
    a: BufferRef[F]      # [nb, m, m] X X^T of the unnormalized X
    inv: BufferRef[F]    # [nb] 1 / Frobenius norm of X
    m: uint32
    nb: uint32


@compute_shader(local_size=(ROW_WG, 1, 1))
def muon_norm(global_id: uvec3, p: MuonNormPush):
    """|X|_F^2 = trace(X X^T), which the first Newton-Schulz product computes anyway."""
    b = flat_row(global_id)
    if b >= p.nb:
        return
    s = 0.0
    base = b * p.m * p.m
    for i in range(p.m):
        s += p.a[base + uint32(i) * (p.m + uint32(1))].v
    p.inv[b].v = 1.0 / (sqrt(s) + 1e-7)


@gpu_struct
class BMMPush:
    """out[b] = alpha sc[b]^pab A[b] B[b] + beta sc[b]^pd D[b], all row-major.

    A is M x K, B is K x N, D and out are M x N. M, N and K are multiples of 4. Dispatch y splits the
    sum over K into chunks of KS vec4s: chunk 0 writes out, chunk s > 0 writes
    part[s - 1] (same shape as out), for ``reduce_partials4`` to add in.
    """
    out: BufferRef[F4]
    a: BufferRef[F4]
    b: BufferRef[F4]
    d: BufferRef[F4]
    sc: BufferRef[F]   # [nb] per-matrix scale (the inverse norm)
    part: BufferRef[F4]
    outT: BufferRef[F4]  # with tout != 0: out transposed as well, [nb, N, M]
    M: uint32
    N: uint32
    K: uint32
    nb: uint32
    tout: uint32
    KS: uint32         # vec4s of K per chunk
    pab: uint32        # power of sc[b] on the product
    pd: uint32         # power of sc[b] on D
    alpha: float32
    beta: float32      # 0: D is not read


@shader_fn(inline=True)
def sc_pow(sc: BufferRef[F], b: uint32, pw: uint32) -> float32:
    s = 1.0
    if pw != uint32(0):
        x = sc[b].v
        for _ in range(pw):
            s = s * x
    return s


@shader_fn(inline=True)
def bmm_value(d: BufferRef[F4], i: uint32, acc: vec4, sab: float32, beta: float32,
              sd: float32) -> vec4:
    v = sab * acc
    if beta != 0.0:
        v += sd * d[i].v
    return v


@compute_shader(local_size=(ROW_WG, 1, 1))
def bmm_nn(global_id: uvec3, p: BMMPush):
    """One 4 x 4 tile of out = A B; neighbouring invocations read the same rows of A."""
    i = global_id.x
    M4 = p.M / uint32(4)
    N4 = p.N / uint32(4)
    K4 = p.K / uint32(4)
    tiles = M4 * N4
    if i >= p.nb * tiles:
        return
    bt = i / tiles
    t = i % tiles
    r0 = (t / N4) * uint32(4)
    c = t % N4
    k0 = global_id.y * p.KS
    a0 = bt * p.M * K4 + r0 * K4 + k0
    bb = bt * p.K * N4 + c + k0 * uint32(4) * N4
    acc0 = vec4(0.0)
    acc1 = vec4(0.0)
    acc2 = vec4(0.0)
    acc3 = vec4(0.0)
    for k in range(min(p.KS, K4 - k0)):
        kk = uint32(k)
        w0 = bb + kk * uint32(4) * N4
        W = mat4(p.b[w0].v, p.b[w0 + N4].v, p.b[w0 + uint32(2) * N4].v,
                 p.b[w0 + uint32(3) * N4].v)
        acc0 += W * p.a[a0 + kk].v
        acc1 += W * p.a[a0 + K4 + kk].v
        acc2 += W * p.a[a0 + uint32(2) * K4 + kk].v
        acc3 += W * p.a[a0 + uint32(3) * K4 + kk].v
    sab = p.alpha * sc_pow(p.sc, bt, p.pab)
    sd = p.beta * sc_pow(p.sc, bt, p.pd)
    o = bt * p.M * N4 + r0 * N4 + c
    v0 = bmm_value(p.d, o, acc0, sab, p.beta, sd)
    v1 = bmm_value(p.d, o + N4, acc1, sab, p.beta, sd)
    v2 = bmm_value(p.d, o + uint32(2) * N4, acc2, sab, p.beta, sd)
    v3 = bmm_value(p.d, o + uint32(3) * N4, acc3, sab, p.beta, sd)
    if global_id.y != uint32(0):
        q = o + (global_id.y - uint32(1)) * p.nb * p.M * N4
        p.part[q].v = v0
        p.part[q + N4].v = v1
        p.part[q + uint32(2) * N4].v = v2
        p.part[q + uint32(3) * N4].v = v3
        return
    p.out[o].v = v0
    p.out[o + N4].v = v1
    p.out[o + uint32(2) * N4].v = v2
    p.out[o + uint32(3) * N4].v = v3
    if p.tout != uint32(0):
        M4 = p.M / uint32(4)
        q = bt * p.N * M4 + c * uint32(4) * M4 + r0 / uint32(4)
        p.outT[q].v = vec4(v0.x, v1.x, v2.x, v3.x)
        p.outT[q + M4].v = vec4(v0.y, v1.y, v2.y, v3.y)
        p.outT[q + uint32(2) * M4].v = vec4(v0.z, v1.z, v2.z, v3.z)
        p.outT[q + uint32(3) * M4].v = vec4(v0.w, v1.w, v2.w, v3.w)


@gpu_struct
class MuonApplyPush:
    params: BufferRef[F]
    x: BufferRef[F]        # the orthogonalized momentum, in the workspace
    idx: BufferRef[U]
    hyper: BufferRef[Hyper]
    scale: float32         # update RMS matching: 0.2 * sqrt(max(rows, cols))
    n: uint32


@compute_shader(local_size=(WG, 1, 1))
def muon_apply(global_id: uvec3, p: MuonApplyPush):
    """W -= lr (scale O + wd W), scattered back from the workspace."""
    j = flat_index(global_id)
    if j >= p.n:
        return
    hp = p.hyper[uint32(0)]
    k = p.idx[j].v
    w = p.params[k].v
    p.params[k].v = w - hp.lr * (p.scale * p.x[j].v + hp.wd * w)


# -- QK-Clip ------------------------------------------------------------------
# MuonClip (Kimi K2, 2025): after each update, every head whose largest
# attention logit in the step exceeded tau has its query and key weights (and
# biases) scaled by sqrt(tau / max logit) each, which scales its logits by
# tau / max logit. Heads at or below tau are left alone.

@gpu_struct
class QKPush:
    smax: BufferRef[F]   # [L, B, H, T] row maxima written by the forward attention
    part: BufferRef[F]   # [L, H, B]
    out: BufferRef[F]    # [L, H]
    prev: BufferRef[F]   # [L, H] maxima of the step's earlier micro-batches
    L: uint32
    H: uint32
    B: uint32
    T: uint32
    accumulate: uint32


@compute_shader(local_size=(ROW_WG, 1, 1))
def qk_max_partial(global_id: uvec3, p: QKPush):
    j = flat_row(global_id)   # (l*H + h)*B + b
    if j >= p.L * p.H * p.B:
        return
    b = j % p.B
    lh = j / p.B
    base = (lh / p.H) * p.B * p.H * p.T + (b * p.H + lh % p.H) * p.T
    mx = -1e30
    for t in range(p.T):
        mx = max(mx, p.smax[base + uint32(t)].v)
    p.part[j].v = mx


@compute_shader(local_size=(ROW_WG, 1, 1))
def qk_max(global_id: uvec3, p: QKPush):
    j = flat_row(global_id)   # l*H + h
    if j >= p.L * p.H:
        return
    mx = -1e30
    for b in range(p.B):
        mx = max(mx, p.part[j * p.B + uint32(b)].v)
    if p.accumulate != uint32(0):
        mx = max(mx, p.prev[j].v)
    p.out[j].v = mx


@gpu_struct
class QKClipPush:
    params: BufferRef[F]
    smax: BufferRef[F]   # [L, H] largest logit of each head in the step
    tau: float32
    off: uint32          # offset of layer 0's qkv parameters
    stride: uint32       # parameters per layer
    L: uint32
    C: uint32
    H: uint32


@compute_shader(local_size=(WG, 1, 1))
def qk_clip(global_id: uvec3, p: QKClipPush):
    """Scale the q and k rows (weights and bias) of every head over tau."""
    i = flat_index(global_id)   # (l, row r < 2C, column c <= C; c = C is the bias)
    per = uint32(2) * p.C * (p.C + uint32(1))
    if i >= p.L * per:
        return
    l = i / per
    rem = i % per
    r = rem / (p.C + uint32(1))
    c = rem % (p.C + uint32(1))
    s = p.smax[l * p.H + (r % p.C) / (p.C / p.H)].v
    if s <= p.tau:
        return
    base = p.off + l * p.stride
    k = base + uint32(3) * p.C * p.C + r
    if c < p.C:
        k = base + r * p.C + c
    p.params[k].v = p.params[k].v * sqrt(p.tau / s)
