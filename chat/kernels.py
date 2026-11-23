"""Compute shaders for the chat model: a Llama-style decoder on vabs compute.

What differs from ``vabs_gpt.kernels`` (whose reductions, gather, gradient norm,
AdamW and Muon kernels are reused unchanged):

- tokens are ``uint32`` ids (vocabulary up to 2^32), not packed bytes;
- RMSNorm with a gain and no bias, instead of LayerNorm;
- linear layers have no bias;
- grouped-query attention: H query heads share Hkv key/value heads, and the
  query and key rows are rotated in place by RoPE after the qkv projection;
- a SwiGLU MLP: ``silu(gate) * up``;
- the token embedding is tied to the output head. Its backward pass from the
  embedding lookup is a scatter over the tokens of the batch, grouped on the host;
- cross entropy over a large vocabulary in three dispatches (chunk maxima and
  sums, per-row log-sum-exp, then the gradient element-wise), with masked
  targets (``IGNORE``) that contribute neither loss nor gradient.

Layouts: activations are row-major ``[N = B*T, width]`` float32 arrays read as
``vec4``. A qkv row is ``[q (C) | k (Ckv) | v (Ckv)]`` with ``Ckv = Hkv * hs``.
The attention matrix of query head (b, h) is ``att[b, h, t, t2]`` with rows
padded to whole ``vec4`` of keys, zero after the diagonal (as in vabs_gpt).
"""
from vabs.shaders import (
    barrier,
    compute_shader,
    coopmat,
    coopmat_load,
    coopmat_muladd,
    coopmat_store,
    dot,
    exp,
    inversesqrt,
    log,
    max,
    min,
    mix,
    shader_fn,
    step,
    workgroup_array,
)
from vabs.structs import (
    BufferRef,
    bf16vec4,
    bfloat16,
    float32,
    gpu_struct,
    mat4,
    uint32,
    uvec3,
    vec4,
)

WG = 256
ROW_WG = 64
IGNORE = 0xFFFFFFFF   # target id that is not scored (prompt tokens, padding): any id >= V
CE_CHUNK = 256        # vec4s of logits per partial maximum and sum


@gpu_struct
class F:
    v: float32


@gpu_struct
class F4:
    v: vec4


@gpu_struct
class U:
    v: uint32


@shader_fn
def flat_index(gid: uvec3) -> uint32:
    return gid.x + gid.y * uint32(8388608)  # WG * SPLIT_GROUPS (vabs_gpt.kernels)


@shader_fn
def flat_row(gid: uvec3) -> uint32:
    return gid.x + gid.y * uint32(2097152)  # ROW_WG * SPLIT_GROUPS


@shader_fn
def hsum(x: vec4) -> float32:
    return dot(x, vec4(1.0))


@shader_fn
def hmax(x: vec4) -> float32:
    return max(max(x.x, x.y), max(x.z, x.w))


@shader_fn
def sigmoid4(x: vec4) -> vec4:
    return 1.0 / (1.0 + exp(-x))


# -- embedding ----------------------------------------------------------------

@gpu_struct
class EmbedPush:
    x: BufferRef[F4]      # [N, C]: forward output, backward input gradient
    wte: BufferRef[F4]    # [V, C]: the table, or its gradient (backward)
    toks: BufferRef[U]    # [N] ids
    order: BufferRef[U]   # backward: the token positions grouped by id
    starts: BufferRef[U]  # backward: [0] = groups, then group g is order[starts[1+g]..]
    ids: BufferRef[U]     # backward: the id of each group
    N: uint32
    C: uint32


@compute_shader(local_size=(WG, 1, 1))
def embed_fwd(global_id: uvec3, p: EmbedPush):
    i = flat_index(global_id)
    C4 = p.C / uint32(4)
    if i >= p.N * C4:
        return
    n = i / C4
    p.x[i].v = p.wte[p.toks[n].v * C4 + i % C4].v


@compute_shader(local_size=(WG, 1, 1))
def embed_bwd(global_id: uvec3, p: EmbedPush):
    """wte_grad[id] += sum of the output gradients at the positions holding id.

    One invocation per (group, 4 channels); groups are distinct ids, so no two
    invocations write the same element.
    """
    i = flat_index(global_id)
    C4 = p.C / uint32(4)
    g = i / C4
    if g >= p.starts[uint32(0)].v:
        return
    c = i % C4
    acc = vec4(0.0)
    j = p.starts[uint32(1) + g].v
    end = p.starts[uint32(2) + g].v
    while j < end:
        acc += p.x[p.order[j].v * C4 + c].v
        j += uint32(1)
    o = p.ids[g].v * C4 + c
    p.wte[o].v = p.wte[o].v + acc


# -- RMSNorm ------------------------------------------------------------------

@gpu_struct
class NormPush:
    out: BufferRef[F4]   # forward: normalized output; backward: input gradient
    inp: BufferRef[F4]   # [N, C] input
    w: BufferRef[F4]     # gain [C]
    rstd: BufferRef[F]   # [N]
    dout: BufferRef[F4]  # backward: output gradient
    part: BufferRef[F4]  # backward: [K, C] partial sums of the gain gradient
    N: uint32
    C: uint32
    R: uint32
    accumulate: uint32
    eps: float32


@compute_shader(local_size=(ROW_WG, 1, 1))
def rmsnorm_fwd(global_id: uvec3, p: NormPush):
    n = flat_row(global_id)
    if n >= p.N:
        return
    C4 = p.C / uint32(4)
    base = n * C4
    s = vec4(0.0)
    for c in range(C4):
        x = p.inp[base + uint32(c)].v
        s += x * x
    r = inversesqrt(hsum(s) / float32(p.C) + p.eps)
    for c in range(C4):
        cc = uint32(c)
        p.out[base + cc].v = p.inp[base + cc].v * r * p.w[cc].v
    p.rstd[n].v = r


@compute_shader(local_size=(ROW_WG, 1, 1))
def rmsnorm_bwd(global_id: uvec3, p: NormPush):
    """dx = r (w dout - xhat mean(w dout xhat)), xhat = x r."""
    n = flat_row(global_id)
    if n >= p.N:
        return
    C4 = p.C / uint32(4)
    base = n * C4
    r = p.rstd[n].v
    s = vec4(0.0)
    for c in range(C4):
        cc = uint32(c)
        s += p.w[cc].v * p.dout[base + cc].v * p.inp[base + cc].v
    k = hsum(s) * r * r / float32(p.C)
    for c in range(C4):
        cc = uint32(c)
        g = r * (p.w[cc].v * p.dout[base + cc].v - p.inp[base + cc].v * k)
        if p.accumulate != uint32(0):
            g += p.out[base + cc].v
        p.out[base + cc].v = g


@compute_shader(local_size=(ROW_WG, 1, 1))
def rmsnorm_bwd_partial(global_id: uvec3, p: NormPush):
    c = global_id.x
    k = global_id.y
    C4 = p.C / uint32(4)
    if c >= C4:
        return
    acc = vec4(0.0)
    for r in range(min(p.R, p.N - k * p.R)):
        n = k * p.R + uint32(r)
        acc += p.dout[n * C4 + c].v * p.inp[n * C4 + c].v * p.rstd[n].v
    p.part[k * C4 + c].v = acc


# -- linear layers (no bias) --------------------------------------------------

@gpu_struct
class MMPush:
    out: BufferRef[F4]   # fwd: [N, OC]; bwd input grad: [N, C]; bwd weights: partials [K, OC*C]
    inp: BufferRef[F4]   # fwd/bwd weights: [N, C]; bwd input grad: dout [N, OC]
    w: BufferRef[F4]     # W[OC, C]; fwd: the transposed copy W^T[C, OC]
    res: BufferRef[F4]   # fwd: residual added to the output (flags 1)
    dout: BufferRef[F4]  # bwd weights: [N, OC]
    N: uint32
    C: uint32
    OC: uint32
    flags: uint32        # 1: add residual
    R: uint32


@compute_shader(local_size=(WG, 1, 1))
def matmul_fwd(global_id: uvec3, p: MMPush):
    """out[n, oc] = inp[n, :] . W[oc, :] (+ res) for 4 rows x 4 outputs (see vabs_gpt)."""
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
    a0 = vec4(0.0)
    a1 = vec4(0.0)
    a2 = vec4(0.0)
    a3 = vec4(0.0)
    for c in range(C4):
        cc = uint32(c)
        w0 = cc * uint32(4) * OC4 + o
        W = mat4(p.w[w0].v, p.w[w0 + OC4].v, p.w[w0 + uint32(2) * OC4].v,
                 p.w[w0 + uint32(3) * OC4].v)
        a0 += W * p.inp[r0 + cc].v
        a1 += W * p.inp[r1 + cc].v
        a2 += W * p.inp[r2 + cc].v
        a3 += W * p.inp[r3 + cc].v
    if (p.flags & uint32(1)) != uint32(0):
        a0 += p.res[n0 * OC4 + o].v
        a1 += p.res[n1 * OC4 + o].v
        a2 += p.res[n2 * OC4 + o].v
        a3 += p.res[n3 * OC4 + o].v
    p.out[n0 * OC4 + o].v = a0
    if n0 + uint32(1) <= last:
        p.out[n1 * OC4 + o].v = a1
    if n0 + uint32(2) <= last:
        p.out[n2 * OC4 + o].v = a2
    if n0 + uint32(3) <= last:
        p.out[n3 * OC4 + o].v = a3


@compute_shader(local_size=(WG, 1, 1))
def matmul_bwd_inp(global_id: uvec3, p: MMPush):
    """out[n, c] = sum_oc inp[n, oc] W[oc, c] for 4 rows x 4 channels."""
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
    p.out[n0 * C4 + c].v = a0
    if n0 + uint32(1) <= last:
        p.out[n1 * C4 + c].v = a1
    if n0 + uint32(2) <= last:
        p.out[n2 * C4 + c].v = a2
    if n0 + uint32(3) <= last:
        p.out[n3 * C4 + c].v = a3


@compute_shader(local_size=(WG, 1, 1))
def matmul_bwd_w_partial(global_id: uvec3, p: MMPush):
    """Partial sums over one chunk of R rows of dW[oc, c] = sum_n dout[n, oc] inp[n, c]."""
    j = global_id.x
    k = global_id.y
    C4 = p.C / uint32(4)
    OC4 = p.OC / uint32(4)
    if j >= OC4 * C4:
        return
    n0 = k * p.R
    rows = min(p.R, p.N - n0)
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
    base = k * OC4 * p.C + o * uint32(4) * C4 + c
    p.out[base].v = a0
    p.out[base + C4].v = a1
    p.out[base + uint32(2) * C4].v = a2
    p.out[base + uint32(3) * C4].v = a3


# -- RoPE ---------------------------------------------------------------------

@gpu_struct
class RopePush:
    x: BufferRef[F4]     # qkv rows, rotated in place (or their gradients)
    cs: BufferRef[F4]    # [T, hs/4]: (cos, sin) of the two pairs in each vec4
    N: uint32
    T: uint32
    S4: uint32           # vec4s per qkv row
    R4: uint32           # vec4s rotated per row: q and k, (H + Hkv) * hs / 4
    hs4: uint32
    inverse_: uint32     # 1: rotate by -theta (the backward pass)


@compute_shader(local_size=(WG, 1, 1))
def rope(global_id: uvec3, p: RopePush):
    """Rotate the pairs (x[2j], x[2j+1]) of every q and k head by t * theta_j."""
    i = flat_index(global_id)
    if i >= p.N * p.R4:
        return
    n = i / p.R4
    c = i % p.R4
    f = p.cs[(n % p.T) * p.hs4 + c % p.hs4].v
    s0 = f.y
    s1 = f.w
    if p.inverse_ != uint32(0):
        s0 = -s0
        s1 = -s1
    o = n * p.S4 + c
    x = p.x[o].v
    p.x[o].v = vec4(x.x * f.x - x.y * s0, x.x * s0 + x.y * f.x,
                    x.z * f.z - x.w * s1, x.z * s1 + x.w * f.z)


# -- grouped-query causal attention -------------------------------------------

@gpu_struct
class AttnPush:
    qkv: BufferRef[F4]
    att: BufferRef[F4]   # softmax probabilities
    datt: BufferRef[F4]  # bwd: d(pre-softmax scores)
    dout: BufferRef[F4]  # bwd: gradient of the attention output [N, C]
    y: BufferRef[F4]     # the attention output [N, C]
    B: uint32
    T: uint32
    H: uint32
    G: uint32            # query heads per key/value head
    hs4: uint32
    S4: uint32           # vec4s per qkv row
    C4: uint32           # vec4s of q per row (= start of k)
    KV4: uint32          # vec4s of k per row (= v starts at C4 + KV4)


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
    scale = inversesqrt(float32(p.hs4 * uint32(4)))
    last = p.T - uint32(1)
    q0 = (b * p.T + t) * p.S4 + h * p.hs4
    kb0 = b * p.T * p.S4 + p.C4 + (h / p.G) * p.hs4
    a0 = row * ((p.T + uint32(3)) / uint32(4))
    blocks = t / uint32(4) + uint32(1)
    ft = vec4(float32(t))
    mx = -1e30
    total = 0.0
    for kb in range(blocks):
        t2 = uint32(kb) * uint32(4)
        s = key_dots(p.qkv, q0, p.qkv, kb0, p.S4, t2, last, p.hs4) * scale
        idx = vec4(float32(t2)) + vec4(0.0, 1.0, 2.0, 3.0)
        s = mix(vec4(-1e30), s, step(idx, ft))
        m = max(mx, hmax(s))
        total = total * exp(mx - m) + hsum(exp(s - m))
        mx = m
    inv = 1.0 / total
    for kb in range(blocks):
        t2 = uint32(kb) * uint32(4)
        s = key_dots(p.qkv, q0, p.qkv, kb0, p.S4, t2, last, p.hs4) * scale
        idx = vec4(float32(t2)) + vec4(0.0, 1.0, 2.0, 3.0)
        s = mix(vec4(-1e30), s, step(idx, ft))
        p.att[a0 + uint32(kb)].v = exp(s - mx) * inv


@compute_shader(local_size=(WG, 1, 1))
def attn_bwd_scores(global_id: uvec3, p: AttnPush):
    """datt = att * (dout[t] . v[t2] - dout[t] . y[t]) * scale, one query row."""
    row = flat_index(global_id)
    if row >= p.B * p.H * p.T:
        return
    t = row % p.T
    bh = row / p.T
    h = bh % p.H
    b = bh / p.H
    C4 = p.H * p.hs4
    scale = inversesqrt(float32(p.hs4 * uint32(4)))
    last = p.T - uint32(1)
    o0 = (b * p.T + t) * C4 + h * p.hs4
    vb0 = b * p.T * p.S4 + p.C4 + p.KV4 + (h / p.G) * p.hs4
    a0 = row * ((p.T + uint32(3)) / uint32(4))
    blocks = t / uint32(4) + uint32(1)
    total = 0.0
    for d in range(p.hs4):
        total += dot(p.dout[o0 + uint32(d)].v, p.y[o0 + uint32(d)].v)
    for kb in range(blocks):
        kk = uint32(kb)
        s = key_dots(p.dout, o0, p.qkv, vb0, p.S4, kk * uint32(4), last, p.hs4)
        p.datt[a0 + kk].v = p.att[a0 + kk].v * (s - total) * scale


@gpu_struct
class AttnMMPush:
    """Products with an attention-shaped matrix s[b, h, t, t2] (lower triangular)."""
    out: BufferRef[F4]   # first output channel of head 0, row 0
    x: BufferRef[F4]     # first input channel of head 0, row 0
    s: BufferRef[F4]
    B: uint32
    T: uint32
    H: uint32            # query heads (heads of s)
    G: uint32            # query heads per key/value head
    hs4: uint32
    xs: uint32           # vec4s per row of x
    os: uint32           # vec4s per row of out


@compute_shader(local_size=(WG, 1, 1))
def attn_mm(global_id: uvec3, p: AttnMMPush):
    """out[t, h] = sum_{t2 <= t} s[h, t, t2] x[t2, h / G]: per query head, x per kv head.

    4 queries x 4 channels per invocation (y = att v, dq = datt k).
    """
    i = flat_index(global_id)   # ((b*H + h)*TB + t/4)*hs4 + c/4
    TB = (p.T + uint32(3)) / uint32(4)
    if i >= p.B * p.H * TB * p.hs4:
        return
    c = i % p.hs4
    j = i / p.hs4
    tb = j % TB
    bh = j / TB
    h = bh % p.H
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
    xb = bT * p.xs + (h / p.G) * p.hs4 + c
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
    ob = bT * p.os + h * p.hs4 + c
    p.out[ob + t0 * p.os].v = a0
    if t0 + uint32(1) <= last:
        p.out[ob + t1 * p.os].v = a1
    if t0 + uint32(2) <= last:
        p.out[ob + t2 * p.os].v = a2
    if t0 + uint32(3) <= last:
        p.out[ob + t3 * p.os].v = a3


@compute_shader(local_size=(WG, 1, 1))
def attn_mm_t(global_id: uvec3, p: AttnMMPush):
    """out[t2, g] = sum_{h in group g} sum_{t >= t2} s[h, t, t2] x[t, h]: per kv head.

    4 keys x 4 channels per invocation (dv = att^T dout, dk = datt^T q).
    """
    i = flat_index(global_id)   # ((b*Hkv + g)*TB + t2/4)*hs4 + c/4
    TB = (p.T + uint32(3)) / uint32(4)
    HK = p.H / p.G
    if i >= p.B * HK * TB * p.hs4:
        return
    c = i % p.hs4
    j = i / p.hs4
    kb = j % TB
    bg = j / TB
    g = bg % HK
    b = bg / HK
    bT = b * p.T
    k0 = kb * uint32(4)
    a0 = vec4(0.0)
    a1 = vec4(0.0)
    a2 = vec4(0.0)
    a3 = vec4(0.0)
    for e in range(p.G):
        h = g * p.G + uint32(e)
        sb = (b * p.H + h) * p.T * TB + kb
        xb = bT * p.xs + h * p.hs4 + c
        for r in range(p.T - k0):   # from the last query down: every invocation on the same row
            tt = p.T - uint32(1) - uint32(r)
            s = p.s[sb + tt * TB].v
            x = p.x[xb + tt * p.xs].v
            a0 += s.x * x
            a1 += s.y * x
            a2 += s.z * x
            a3 += s.w * x
    ob = (bT + k0) * p.os + g * p.hs4 + c
    p.out[ob].v = a0
    if k0 + uint32(1) < p.T:
        p.out[ob + p.os].v = a1
    if k0 + uint32(2) < p.T:
        p.out[ob + uint32(2) * p.os].v = a2
    if k0 + uint32(3) < p.T:
        p.out[ob + uint32(3) * p.os].v = a3


# -- SwiGLU -------------------------------------------------------------------

@gpu_struct
class GluPush:
    pre: BufferRef[F4]   # [N, 2F]: gate | up
    h: BufferRef[F4]     # [N, F]: forward output, backward its gradient
    dpre: BufferRef[F4]  # backward: [N, 2F]
    N: uint32
    F: uint32


@compute_shader(local_size=(WG, 1, 1))
def swiglu_fwd(global_id: uvec3, p: GluPush):
    i = flat_index(global_id)
    F4_ = p.F / uint32(4)
    if i >= p.N * F4_:
        return
    n = i / F4_
    f = i % F4_
    g = p.pre[n * uint32(2) * F4_ + f].v
    u = p.pre[n * uint32(2) * F4_ + F4_ + f].v
    p.h[i].v = g * sigmoid4(g) * u


@compute_shader(local_size=(WG, 1, 1))
def swiglu_bwd(global_id: uvec3, p: GluPush):
    i = flat_index(global_id)
    F4_ = p.F / uint32(4)
    if i >= p.N * F4_:
        return
    n = i / F4_
    f = i % F4_
    gi = n * uint32(2) * F4_ + f
    g = p.pre[gi].v
    u = p.pre[gi + F4_].v
    d = p.h[i].v
    sg = sigmoid4(g)
    p.dpre[gi].v = d * u * sg * (1.0 + g * (1.0 - sg))
    p.dpre[gi + F4_].v = d * g * sg


# -- cross entropy ------------------------------------------------------------

@gpu_struct
class CEPush:
    logits: BufferRef[F4]  # [N, V]; replaced by d(loss)/d(logits) by ce_grad
    logit1: BufferRef[F]
    part: BufferRef[F]     # [N, chunks, 2]: chunk maximum and sum of exp
    lse: BufferRef[F]      # [N] log-sum-exp
    losses: BufferRef[F]   # [N] per-token loss (0 where the target is IGNORE)
    targets: BufferRef[U]  # [N]
    scale: BufferRef[F]    # d(loss)/d(per-token loss): [1] 1 / scored tokens in the step,
    N: uint32              # or [N] a weight per token (wstride 1)
    V: uint32
    chunks: uint32
    wstride: uint32


@compute_shader(local_size=(WG, 1, 1))
def ce_partial(global_id: uvec3, p: CEPush):
    i = flat_index(global_id)   # n * chunks + chunk
    if i >= p.N * p.chunks:
        return
    n = i / p.chunks
    k = i % p.chunks
    V4 = p.V / uint32(4)
    v0 = k * uint32(256)   # CE_CHUNK
    v1 = min(v0 + uint32(256), V4)
    base = n * V4
    mx = -1e30
    for v in range(v1 - v0):
        mx = max(mx, hmax(p.logits[base + v0 + uint32(v)].v))
    total = 0.0
    for v in range(v1 - v0):
        total += hsum(exp(p.logits[base + v0 + uint32(v)].v - mx))
    p.part[uint32(2) * i].v = mx
    p.part[uint32(2) * i + uint32(1)].v = total


@compute_shader(local_size=(ROW_WG, 1, 1))
def ce_finish(global_id: uvec3, p: CEPush):
    n = flat_row(global_id)
    if n >= p.N:
        return
    base = uint32(2) * n * p.chunks
    mx = -1e30
    for k in range(p.chunks):
        mx = max(mx, p.part[base + uint32(2) * uint32(k)].v)
    total = 0.0
    for k in range(p.chunks):
        kk = uint32(2) * uint32(k)
        total += p.part[base + kk + uint32(1)].v * exp(p.part[base + kk].v - mx)
    lse = log(total) + mx
    p.lse[n].v = lse
    t = p.targets[n].v
    loss = 0.0
    if t < p.V:   # ids >= V (IGNORE) are not scored
        loss = lse - p.logit1[n * p.V + t].v
    p.losses[n].v = loss


@compute_shader(local_size=(WG, 1, 1))
def ce_grad(global_id: uvec3, p: CEPush):
    """d(loss)/d(logits) = scale (softmax - onehot(target)), 0 for ignored rows.

    With a weight per token (``wstride`` 1) the loss is sum_n w_n (-log p(target_n)): the
    policy-gradient loss of reinforcement learning when w_n is the token's advantage.
    """
    i = flat_index(global_id)
    V4 = p.V / uint32(4)
    if i >= p.N * V4:
        return
    n = i / V4
    t = p.targets[n].v
    if t >= p.V:
        p.logits[i].v = vec4(0.0)
        return
    sc = p.scale[n * p.wstride].v
    g = exp(p.logits[i].v - p.lse[n].v) * sc
    v = (i % V4) * uint32(4)
    if t >= v:
        if t < v + uint32(4):
            idx = vec4(float32(v)) + vec4(0.0, 1.0, 2.0, 3.0)
            ft = vec4(float32(t))
            g -= sc * step(idx, ft) * step(ft, idx)
    p.logits[i].v = g


# -- inference ----------------------------------------------------------------

@gpu_struct
class RowsPush:
    out: BufferRef[F4]   # [B, C]
    src: BufferRef[F4]   # [B*T, C]
    pos: BufferRef[U]    # [B] row within each sequence
    B: uint32
    T: uint32
    C: uint32


@compute_shader(local_size=(WG, 1, 1))
def gather_rows(global_id: uvec3, p: RowsPush):
    """out[b] = src[b*T + pos[b]]: the one position per sequence whose logits are needed."""
    i = flat_index(global_id)
    C4 = p.C / uint32(4)
    if i >= p.B * C4:
        return
    b = i / C4
    p.out[i].v = p.src[(b * p.T + p.pos[b].v) * C4 + i % C4].v


# -- incremental decoding with a key/value cache ----------------------------------
# The cache of layer l is the qkv buffer [B, T, S] of a prefill forward over the
# prompts: a decode step computes the qkv row of one new token per sequence,
# rotates it at the sequence's position pos[b] and stores it in row (b, pos[b]),
# then attends over rows 0..pos[b] of its sequence.

@gpu_struct
class DecodePush:
    x: BufferRef[F4]       # [B, S] new qkv rows (rope_store) / [B, C] output (decode_av)
    cache: BufferRef[F4]   # [B, T, S] the layer's qkv rows
    cs: BufferRef[F4]      # [T, hs/4] rope table
    sc: BufferRef[F]       # [B, H, T] attention scores
    pos: BufferRef[U]      # [B] position of the new token
    B: uint32
    T: uint32
    H: uint32
    G: uint32
    hs4: uint32
    S4: uint32
    C4: uint32
    KV4: uint32


@compute_shader(local_size=(WG, 1, 1))
def rope_store(global_id: uvec3, p: DecodePush):
    """cache[b, pos[b]] = x[b] with its q and k heads rotated at pos[b]."""
    i = flat_index(global_id)
    if i >= p.B * p.S4:
        return
    b = i / p.S4
    c = i % p.S4
    t = p.pos[b].v
    x = p.x[i].v
    if c < p.C4 + p.KV4:
        f = p.cs[t * p.hs4 + c % p.hs4].v
        x = vec4(x.x * f.x - x.y * f.y, x.x * f.y + x.y * f.x,
                 x.z * f.z - x.w * f.w, x.z * f.w + x.w * f.z)
    p.cache[(b * p.T + t) * p.S4 + c].v = x


@compute_shader(local_size=(WG, 1, 1))
def decode_scores(global_id: uvec3, p: DecodePush):
    """sc[b, h, t] = q[b, h] . k[b, t, h / G] / sqrt(hs) for t <= pos[b]."""
    i = flat_index(global_id)   # (b*H + h)*T + t
    if i >= p.B * p.H * p.T:
        return
    t = i % p.T
    bh = i / p.T
    h = bh % p.H
    b = bh / p.H
    if t > p.pos[b].v:
        return
    q0 = (b * p.T + p.pos[b].v) * p.S4 + h * p.hs4
    k0 = (b * p.T + t) * p.S4 + p.C4 + (h / p.G) * p.hs4
    s = 0.0
    for d in range(p.hs4):
        s += dot(p.cache[q0 + uint32(d)].v, p.cache[k0 + uint32(d)].v)
    p.sc[i].v = s * inversesqrt(float32(p.hs4 * uint32(4)))


@compute_shader(local_size=(ROW_WG, 1, 1))
def decode_av(global_id: uvec3, p: DecodePush):
    """x[b, h, c] = sum_t softmax(sc[b, h, :pos+1])[t] v[b, t, h / G, c], 4 channels each."""
    i = flat_row(global_id)   # (b*H + h)*hs4 + c
    if i >= p.B * p.H * p.hs4:
        return
    c = i % p.hs4
    bh = i / p.hs4
    h = bh % p.H
    b = bh / p.H
    n = p.pos[b].v + uint32(1)
    s0 = bh * p.T
    mx = -1e30
    for t in range(n):
        mx = max(mx, p.sc[s0 + uint32(t)].v)
    total = 0.0
    acc = vec4(0.0)
    v0 = b * p.T * p.S4 + p.C4 + p.KV4 + (h / p.G) * p.hs4 + c
    for t in range(n):
        e = exp(p.sc[s0 + uint32(t)].v - mx)
        total += e
        acc += e * p.cache[v0 + uint32(t) * p.S4].v
    p.x[bh * p.hs4 + c].v = acc / total


@shader_fn
def hash32(x: uint32) -> uint32:
    h = x ^ (x >> uint32(16))
    h = h * uint32(0x7feb352d)
    h = h ^ (h >> uint32(15))
    h = h * uint32(0x68e31da5)
    return h ^ (h >> uint32(16))


@gpu_struct
class SamplePush:
    logits: BufferRef[F4]  # [B, V]
    part: BufferRef[F]     # [B, chunks, 2]: best perturbed logit of the chunk and its id
    knobs: BufferRef[F]    # [0] 1 / temperature, [1] 1 to add Gumbel noise, 0 for greedy
    seed: BufferRef[U]     # [0]
    tok: BufferRef[U]      # [B] the sampled ids (the next step's input)
    pos: BufferRef[U]      # [B] advanced by one, at most T - 1
    hist: BufferRef[U]     # [steps, B] every step's ids
    B: uint32
    V: uint32
    chunks: uint32
    T: uint32
    step_: uint32


@compute_shader(local_size=(WG, 1, 1))
def sample_partial(global_id: uvec3, p: SamplePush):
    """Gumbel-max over a chunk: argmax_v logits[v] / temperature + g_v, ties to the lower id."""
    i = flat_index(global_id)   # b * chunks + chunk
    if i >= p.B * p.chunks:
        return
    b = i / p.chunks
    k = i % p.chunks
    V4 = p.V / uint32(4)
    v0 = k * uint32(256)   # CE_CHUNK
    v1 = min(v0 + uint32(256), V4)
    inv_t = p.knobs[uint32(0)].v
    noise = p.knobs[uint32(1)].v
    key = hash32(p.seed[uint32(0)].v ^ hash32(b * uint32(7919) + p.step_ * uint32(104729)))
    best = -3e38
    arg = 0.0
    for v in range(v1 - v0):
        vv = v0 + uint32(v)
        l = p.logits[b * V4 + vv].v * inv_t
        e = vv * uint32(4)
        u = vec4(float32(hash32(key + e) >> uint32(8)), float32(hash32(key + e + uint32(1)) >> uint32(8)),
                 float32(hash32(key + e + uint32(2)) >> uint32(8)),
                 float32(hash32(key + e + uint32(3)) >> uint32(8)))
        u = (u + 0.5) * (1.0 / 16777216.0)
        l = l - noise * log(-log(u))
        fe = float32(e)
        if l.x > best:
            best = l.x
            arg = fe
        if l.y > best:
            best = l.y
            arg = fe + 1.0
        if l.z > best:
            best = l.z
            arg = fe + 2.0
        if l.w > best:
            best = l.w
            arg = fe + 3.0
    p.part[uint32(2) * i].v = best
    p.part[uint32(2) * i + uint32(1)].v = arg


@compute_shader(local_size=(ROW_WG, 1, 1))
def sample_finish(global_id: uvec3, p: SamplePush):
    b = flat_row(global_id)
    if b >= p.B:
        return
    best = -3e38
    arg = 0.0
    for k in range(p.chunks):
        j = uint32(2) * (b * p.chunks + uint32(k))
        if p.part[j].v > best:
            best = p.part[j].v
            arg = p.part[j + uint32(1)].v
    t = uint32(arg)
    p.tok[b].v = t
    p.hist[p.step_ * p.B + b].v = t
    p.pos[b].v = min(p.pos[b].v + uint32(1), p.T - uint32(1))


# -- tensor-core matrix products (bfloat16 cooperative matrices) ---------------------
#
# ``mm_coop`` computes C = A B (+ residual) with bfloat16 operands and float32
# accumulation on the tensor cores. A and B stay float32 in memory: each 64 x 64
# output tile stages 32-wide slices of A and B into workgroup memory, rounding them
# to bfloat16, and its four subgroups each multiply a 32 x 32 quarter as 2 x 2
# accumulators of 16 x 16. Flags give the memory layout of each operand, so the
# one kernel covers all three products of a linear layer:
#
#   forward        out[N, OC] = x[N, C] W^T     A row-major, B = W column-major
#   input grad     dx[N, C]   = dy[N, OC] W     A row-major, B row-major
#   weight grad    dW[OC, C]  = dy^T x          A = dy column-major, B row-major,
#                                               split over row chunks
#
# and, with a batch of matrices and the scaled epilogue (CM_SCALE), the products of
# Muon's Newton-Schulz iteration: C = alpha s^pab A B + beta s^pd D, s a per-matrix
# scale. Workgroup z enumerates (matrix, reduction chunk) pairs. A matrix index i
# splits into (i / nh, i % nh) -- (sequence, head) for attention -- and operand X
# of matrix i starts at (i / nh) sX + ((i % nh) / gX) shX floats, which maps the
# query heads of a group onto their shared key/value head (gX = G). The causal
# flags bound the reduction of products with a triangular operand.
#
# All of M, N and K must be multiples of 16 (and the leading dimensions and batch
# strides of 4). The staged tiles are aligned to 16 bytes, as Vulkan requires of
# cooperative-matrix loads; rows are padded so that consecutive rows start in
# different banks.

CM_BM = 64          # output rows per workgroup
CM_BN = 64          # output columns per workgroup
CM_BK = 32          # reduction slice staged per step
CM_LDK = 10         # bf16vec4 per staged row of CM_BK values (8 used, 16-byte multiple)
CM_LDW = 18         # bf16vec4 per staged row of 64 values (16 used)
CM_WG = 128         # four subgroups of 32

CM_A_COLUMN_MAJOR = 1
CM_B_COLUMN_MAJOR = 2
CM_RESIDUAL = 4     # C starts from res (same layout as C)
CM_SCALE = 8        # C = alpha s^pab A B (+ beta s^pd D when beta != 0)
CM_CAUSAL_END = 16      # A is lower triangular: the reduction stops at the tile's last row
CM_CAUSAL_BEGIN = 32    # A is upper triangular: the reduction starts at the tile's first row
CM_SKIP_ABOVE = 64      # leave tiles entirely above the diagonal unwritten

CM_TILE_A = workgroup_array(bf16vec4, 64 * CM_LDW)
CM_TILE_B = workgroup_array(bf16vec4, 64 * CM_LDW)
CmA = coopmat(bfloat16, 16, 16, 'a')
CmB = coopmat(bfloat16, 16, 16, 'b')
CmC = coopmat(float32, 16, 16, 'acc')


@gpu_struct
class F1:
    v: float32


@gpu_struct
class MMCPush:
    a: BufferRef[F4]     # A, row-major [M, K] or column-major (stored [K, M])
    b: BufferRef[F4]     # B, row-major [K, N] or column-major (stored [N, K])
    c: BufferRef[F1]     # C [M, N] (each reduction chunk z at c + z * split)
    res: BufferRef[F1]   # residual added to C (CM_RESIDUAL), same layout
    d: BufferRef[F1]     # D of the scaled epilogue, same layout as C
    sc: BufferRef[F1]    # [nb] per-matrix scales s
    M: uint32
    N: uint32
    K: uint32
    lda: uint32          # floats between rows of A as stored
    ldb: uint32
    ldc: uint32
    flags: uint32
    KC: uint32           # reduction length per chunk (a multiple of CM_BK)
    split: uint32        # floats between the outputs of consecutive chunks
    sa: uint32           # floats between consecutive matrices of A, B and C (and D, res)
    sb: uint32
    scs: uint32
    nh: uint32           # inner matrix count (heads); 1 for a plain batch
    ga: uint32           # inner index divisors and strides per operand (see above)
    gb: uint32
    gc: uint32
    sha: uint32
    shb: uint32
    shc: uint32
    pab: uint32          # powers of s on the product and on D
    pd: uint32
    alpha: float32
    beta: float32


@shader_fn(inline=True)
def cm_scale(sc: BufferRef[F1], b: uint32, pw: uint32, factor: float32) -> float32:
    s = factor
    if pw != uint32(0):
        x = sc[b].v
        for _ in range(pw):
            s = s * x
    return s


@compute_shader(local_size=(CM_WG, 1, 1))
def mm_coop(workgroup_id: uvec3, local_index: uint32, subgroup_id: uint32, p: MMCPush):
    chunks = (p.K + p.KC - uint32(1)) / p.KC
    bt = workgroup_id.z / chunks
    chunk = workgroup_id.z % chunks
    m0 = workgroup_id.y * uint32(CM_BM)
    n0 = workgroup_id.x * uint32(CM_BN)
    if (p.flags & uint32(CM_SKIP_ABOVE)) != uint32(0) and n0 >= m0 + uint32(CM_BM):
        return
    kbeg = chunk * p.KC
    kend = min(p.K, kbeg + p.KC)
    if (p.flags & uint32(CM_CAUSAL_END)) != uint32(0):
        kend = min(kend, m0 + uint32(CM_BM))
    if (p.flags & uint32(CM_CAUSAL_BEGIN)) != uint32(0):
        kbeg = max(kbeg, m0)
    outer = bt / p.nh
    inner = bt % p.nh
    a_col = (p.flags & uint32(CM_A_COLUMN_MAJOR)) != uint32(0)
    b_col = (p.flags & uint32(CM_B_COLUMN_MAJOR)) != uint32(0)
    lda4 = p.lda / uint32(4)
    ldb4 = p.ldb / uint32(4)
    a4 = (outer * p.sa + (inner / p.ga) * p.sha) / uint32(4)
    b4 = (outer * p.sb + (inner / p.gb) * p.shb) / uint32(4)
    sm = (subgroup_id / uint32(2)) * uint32(32)    # the 32 x 32 quarter of this subgroup
    sn = (subgroup_id % uint32(2)) * uint32(32)
    r0 = m0 + sm
    c0 = n0 + sn
    r1 = r0 + uint32(16)
    c1 = c0 + uint32(16)
    in00 = r0 < p.M and c0 < p.N
    in01 = r0 < p.M and c1 < p.N
    in10 = r1 < p.M and c0 < p.N
    in11 = r1 < p.M and c1 < p.N
    base = outer * p.scs + (inner / p.gc) * p.shc  # this matrix within C, D and res
    o00 = base + r0 * p.ldc + c0
    o01 = base + r0 * p.ldc + c1
    o10 = base + r1 * p.ldc + c0
    o11 = base + r1 * p.ldc + c1
    acc00 = CmC(0.0)
    acc01 = CmC(0.0)
    acc10 = CmC(0.0)
    acc11 = CmC(0.0)
    if (p.flags & uint32(CM_RESIDUAL)) != uint32(0):
        if in00:
            acc00 = coopmat_load(CmC, p.res[o00].v, p.ldc)
        if in01:
            acc01 = coopmat_load(CmC, p.res[o01].v, p.ldc)
        if in10:
            acc10 = coopmat_load(CmC, p.res[o10].v, p.ldc)
        if in11:
            acc11 = coopmat_load(CmC, p.res[o11].v, p.ldc)
    k0 = kbeg
    while k0 < kend:
        for j in range(4):
            t = local_index + uint32(j * CM_WG)
            # A slice: 64 rows x 32 values (row-major) or 32 rows x 64 values (column-major)
            v = vec4(0.0)
            if a_col:
                k = k0 + t / uint32(16)
                m = m0 + (t % uint32(16)) * uint32(4)
                if k < kend and m < p.M:
                    v = p.a[a4 + k * lda4 + m / uint32(4)].v
                CM_TILE_A[(t / uint32(16)) * uint32(CM_LDW) + t % uint32(16)] = bf16vec4(v)
            else:
                m = m0 + t / uint32(8)
                k = k0 + (t % uint32(8)) * uint32(4)
                if k < kend and m < p.M:
                    v = p.a[a4 + m * lda4 + k / uint32(4)].v
                CM_TILE_A[(t / uint32(8)) * uint32(CM_LDK) + t % uint32(8)] = bf16vec4(v)
            # B slice: 32 rows x 64 values (row-major) or 64 rows x 32 values (column-major)
            w = vec4(0.0)
            if b_col:
                n = n0 + t / uint32(8)
                k = k0 + (t % uint32(8)) * uint32(4)
                if k < kend and n < p.N:
                    w = p.b[b4 + n * ldb4 + k / uint32(4)].v
                CM_TILE_B[(t / uint32(8)) * uint32(CM_LDK) + t % uint32(8)] = bf16vec4(w)
            else:
                k = k0 + t / uint32(16)
                n = n0 + (t % uint32(16)) * uint32(4)
                if k < kend and n < p.N:
                    w = p.b[b4 + k * ldb4 + n / uint32(4)].v
                CM_TILE_B[(t / uint32(16)) * uint32(CM_LDW) + t % uint32(16)] = bf16vec4(w)
        barrier()
        for kk in range(2):
            ks = uint32(kk * 16)
            if a_col:
                a0 = coopmat_load(CmA, CM_TILE_A[ks * uint32(CM_LDW) + sm / uint32(4)],
                                  uint32(CM_LDW), True)
                a1 = coopmat_load(CmA, CM_TILE_A[ks * uint32(CM_LDW) + sm / uint32(4)
                                                 + uint32(4)], uint32(CM_LDW), True)
            else:
                a0 = coopmat_load(CmA, CM_TILE_A[sm * uint32(CM_LDK) + ks / uint32(4)],
                                  uint32(CM_LDK))
                a1 = coopmat_load(CmA, CM_TILE_A[(sm + uint32(16)) * uint32(CM_LDK)
                                                 + ks / uint32(4)], uint32(CM_LDK))
            if b_col:
                b0 = coopmat_load(CmB, CM_TILE_B[sn * uint32(CM_LDK) + ks / uint32(4)],
                                  uint32(CM_LDK), True)
                b1 = coopmat_load(CmB, CM_TILE_B[(sn + uint32(16)) * uint32(CM_LDK)
                                                 + ks / uint32(4)], uint32(CM_LDK), True)
            else:
                b0 = coopmat_load(CmB, CM_TILE_B[ks * uint32(CM_LDW) + sn / uint32(4)],
                                  uint32(CM_LDW))
                b1 = coopmat_load(CmB, CM_TILE_B[ks * uint32(CM_LDW) + sn / uint32(4)
                                                 + uint32(4)], uint32(CM_LDW))
            acc00 = coopmat_muladd(a0, b0, acc00)
            acc01 = coopmat_muladd(a0, b1, acc01)
            acc10 = coopmat_muladd(a1, b0, acc10)
            acc11 = coopmat_muladd(a1, b1, acc11)
        barrier()
        k0 += uint32(CM_BK)
    if (p.flags & uint32(CM_SCALE)) != uint32(0):
        sab = cm_scale(p.sc, bt, p.pab, p.alpha)
        acc00 = acc00 * sab
        acc01 = acc01 * sab
        acc10 = acc10 * sab
        acc11 = acc11 * sab
        if p.beta != 0.0:
            sd = cm_scale(p.sc, bt, p.pd, p.beta)
            if in00:
                acc00 = acc00 + coopmat_load(CmC, p.d[o00].v, p.ldc) * sd
            if in01:
                acc01 = acc01 + coopmat_load(CmC, p.d[o01].v, p.ldc) * sd
            if in10:
                acc10 = acc10 + coopmat_load(CmC, p.d[o10].v, p.ldc) * sd
            if in11:
                acc11 = acc11 + coopmat_load(CmC, p.d[o11].v, p.ldc) * sd
    out0 = chunk * p.split
    if in00:
        coopmat_store(acc00, p.c[out0 + o00].v, p.ldc)
    if in01:
        coopmat_store(acc01, p.c[out0 + o01].v, p.ldc)
    if in10:
        coopmat_store(acc10, p.c[out0 + o10].v, p.ldc)
    if in11:
        coopmat_store(acc11, p.c[out0 + o11].v, p.ldc)


# -- attention rows around the tensor-core products ----------------------------------
#
# With bf16 precision the attention products run through ``mm_coop``; these kernels
# do the row-wise parts. One workgroup of ROWS_WG handles one row (b, h, t) of the
# [B, H, T, T] score matrix, which keeps its reads and writes contiguous, and
# reduces through workgroup memory.

ROWS_WG = 64
ROW_RED = workgroup_array(float32, ROWS_WG)
SPLIT_ROWS = 32768   # vabs_gpt.kernels.SPLIT_GROUPS: a 1-D dispatch wraps into y rows


@gpu_struct
class AttnRowsPush:
    att: BufferRef[F4]   # [B, H, T, T]: in raw scores, out probabilities (fwd);
    #                      probabilities (bwd)
    datt: BufferRef[F4]  # bwd: in dout . v, out the score gradient
    dout: BufferRef[F4]  # bwd: gradient of the attention output [N, C]
    y: BufferRef[F4]     # bwd: the attention output [N, C]
    rows: uint32         # B * H * T
    T: uint32
    H: uint32
    hs4: uint32


@shader_fn(inline=True)
def row_reduce_max(local_index: uint32, x: float32) -> float32:
    ROW_RED[local_index] = x
    barrier()
    for k in range(6):   # 64 -> 1
        step_ = uint32(32) >> uint32(k)
        if local_index < step_:
            ROW_RED[local_index] = max(ROW_RED[local_index], ROW_RED[local_index + step_])
        barrier()
    r = ROW_RED[0]
    barrier()
    return r


@shader_fn(inline=True)
def row_reduce_sum(local_index: uint32, x: float32) -> float32:
    ROW_RED[local_index] = x
    barrier()
    for k in range(6):
        step_ = uint32(32) >> uint32(k)
        if local_index < step_:
            ROW_RED[local_index] = ROW_RED[local_index] + ROW_RED[local_index + step_]
        barrier()
    r = ROW_RED[0]
    barrier()
    return r


@compute_shader(local_size=(ROWS_WG, 1, 1))
def attn_softmax_rows(workgroup_id: uvec3, local_index: uint32, p: AttnRowsPush):
    """att[t, :] = softmax over t2 <= t of scores[t, t2] / sqrt(hs); zero after t."""
    row = workgroup_id.x + workgroup_id.y * uint32(SPLIT_ROWS)
    if row >= p.rows:
        return
    t = row % p.T
    T4 = p.T / uint32(4)
    base = row * T4
    blocks = t / uint32(4) + uint32(1)
    scale = inversesqrt(float32(p.hs4 * uint32(4)))
    ft = vec4(float32(t))
    mx = -1e30
    j = local_index
    while j < blocks:
        idx = vec4(float32(j * uint32(4))) + vec4(0.0, 1.0, 2.0, 3.0)
        s = mix(vec4(-1e30), p.att[base + j].v * scale, step(idx, ft))
        mx = max(mx, hmax(s))
        j += uint32(ROWS_WG)
    mx = row_reduce_max(local_index, mx)
    total = 0.0
    j = local_index
    while j < blocks:
        idx = vec4(float32(j * uint32(4))) + vec4(0.0, 1.0, 2.0, 3.0)
        s = mix(vec4(-1e30), p.att[base + j].v * scale, step(idx, ft))
        total += hsum(exp(s - mx))
        j += uint32(ROWS_WG)
    inv = 1.0 / row_reduce_sum(local_index, total)
    j = local_index
    while j < T4:
        v = vec4(0.0)
        if j < blocks:
            idx = vec4(float32(j * uint32(4))) + vec4(0.0, 1.0, 2.0, 3.0)
            s = mix(vec4(-1e30), p.att[base + j].v * scale, step(idx, ft))
            v = exp(s - mx) * inv
        p.att[base + j].v = v
        j += uint32(ROWS_WG)


@compute_shader(local_size=(ROWS_WG, 1, 1))
def attn_bwd_rows(workgroup_id: uvec3, local_index: uint32, p: AttnRowsPush):
    """datt = att * (datt - dout[t] . y[t]) * scale, where datt holds dout . v; zero after t."""
    row = workgroup_id.x + workgroup_id.y * uint32(SPLIT_ROWS)
    if row >= p.rows:
        return
    t = row % p.T
    bh = row / p.T
    h = bh % p.H
    b = bh / p.H
    T4 = p.T / uint32(4)
    base = row * T4
    blocks = t / uint32(4) + uint32(1)
    o0 = (b * p.T + t) * p.H * p.hs4 + h * p.hs4
    part = 0.0
    if local_index < p.hs4:
        part = dot(p.dout[o0 + local_index].v, p.y[o0 + local_index].v)
    total = row_reduce_sum(local_index, part)
    scale = inversesqrt(float32(p.hs4 * uint32(4)))
    j = local_index
    while j < T4:
        v = vec4(0.0)
        if j < blocks:
            v = p.att[base + j].v * (p.datt[base + j].v - total) * scale
        p.datt[base + j].v = v
        j += uint32(ROWS_WG)
