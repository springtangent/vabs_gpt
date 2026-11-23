"""Time the chat model's linear-layer products: float32 kernels vs bfloat16 tensor cores.

    python -m vabs_gpt.chat.bench_mm [--tokens 8192] [--dim 512] [--mlp 1344] [--vocab 20000]

For every product of a training step (forward, input gradient, weight gradient
of each linear layer shape) prints the fastest of many GPU timestamp intervals
for the float32 kernel and for ``mm_coop``, and the effective TFLOP/s. Values in
the buffers are arbitrary; only the shapes matter.
"""
import argparse
import array

import vk
from vabs.memory import MemoryUsage

from vabs_gpt.gpu import ComputeContext, Recording, compute_barrier

from . import kernels as K
from .model import PARTIAL_FLOATS, Device, _ceil, coop_splits, linear_rows


def products(tokens, C, F, V, S):
    """``(name, M, N, K)`` for out[M, N] = in[M, K] W^T, per linear layer shape."""
    return [('qkv', tokens, S, C), ('proj', tokens, C, C), ('fc', tokens, 2 * F, C),
            ('down', tokens, C, F), ('head', tokens, V, C)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=8192)
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--mlp', type=int, default=1344)
    ap.add_argument('--vocab', type=int, default=20000)
    ap.add_argument('--kv-dim', type=int, default=256)
    ap.add_argument('--rounds', type=int, default=50)
    args = ap.parse_args()
    S = args.dim + 2 * args.kv_dim
    shapes = products(args.tokens, args.dim, args.mlp, args.vocab, S)
    biggest = max(M * N for _, M, N, _ in shapes)
    with ComputeContext() as ctx:
        if not ctx.supports_bf16_matmul:
            raise SystemExit('this device has no bfloat16 cooperative matrices')
        dev = Device(ctx)
        n_in = max(M * Kd for _, M, _, Kd in shapes)
        n_w = max(N * Kd for _, _, N, Kd in shapes)
        x, w, y, part = (r.device_address for r in dev.arena(
            [4 * n_in, 4 * n_w, 4 * biggest, 4 * PARTIAL_FLOATS], MemoryUsage.DEVICE_LOCAL))
        cases = []   # (label, flops, record(frame))
        for name, N, OC, C in shapes:
            flops = 2 * N * OC * C
            N4 = _ceil(N, 4)
            fwd = K.MMPush(out=y, inp=x, w=w, N=N, C=C, OC=OC)
            cases.append((f'{name} fwd   f32', flops,
                          lambda f, p=fwd, n=N4 * OC // 4: dev.run(f, 'matmul_fwd', p, n)))
            cases.append((f'{name} fwd   bf16', flops,
                          lambda f, N=N, OC=OC, C=C: dev.mm_coop(f, y, x, w, N, OC, C,
                                                                 b_col=True)))
            inp = K.MMPush(out=x, inp=y, w=w, N=N, C=C, OC=OC)
            cases.append((f'{name} dx    f32', flops,
                          lambda f, p=inp, n=N4 * C // 4: dev.run(f, 'matmul_bwd_inp', p, n)))
            cases.append((f'{name} dx    bf16', flops,
                          lambda f, N=N, OC=OC, C=C: dev.mm_coop(f, x, y, w, N, C, OC)))
            R = linear_rows(N, OC * C)
            wgt = K.MMPush(out=part, inp=x, dout=y, N=N, C=C, OC=OC, R=R)

            def f32_w(f, p=wgt, M=OC * C, R=R, N=N, OC=OC, C=C):
                k = _ceil(N, R)
                dev.run(f, 'matmul_bwd_w_partial', p, (OC // 4) * (C // 4), k)
                from vabs_gpt import kernels as GK
                dev.run(f, 'reduce_partials4', GK.Reduce4Push(out=w, part=part, M4=M // 4, K=k),
                        M // 4)

            def bf16_w(f, N=N, OC=OC, C=C):
                splits, kc = coop_splits(OC, C, N, PARTIAL_FLOATS)
                dev.mm_coop(f, part, y, x, OC, C, N, a_col=True, kc=kc)
                from vabs_gpt import kernels as GK
                dev.run(f, 'reduce_partials4', GK.Reduce4Push(out=w, part=part,
                                                              M4=OC * C // 4, K=splits),
                        OC * C // 4)

            cases.append((f'{name} dW    f32', flops, f32_w))
            cases.append((f'{name} dW    bf16', flops, bf16_w))
        count = len(cases) + 1
        pool = vk.create_query_pool(ctx.device, vk.QueryPoolCreateInfo(
            query_type=vk.QueryType.TIMESTAMP, query_count=count))
        stamps, = dev.arena([8 * count], MemoryUsage.HOST_VISIBLE)
        rec = Recording(ctx)
        with rec.record() as frame:
            cmd = rec._cmd
            vk.cmd_reset_query_pool(cmd, pool, 0, count)
            vk.cmd_write_timestamp2(cmd, vk.PipelineStageFlags2.ALL_COMMANDS, pool, 0)
            for i, (_, _, record) in enumerate(cases):
                record(frame)
                compute_barrier(frame)
                vk.cmd_write_timestamp2(cmd, vk.PipelineStageFlags2.ALL_COMMANDS, pool, i + 1)
            vk.cmd_copy_query_pool_results(cmd, pool, 0, count, stamps.arena.buffer_handle,
                                           stamps.offset, 8, vk.QueryResultFlags.RESULT_64
                                           | vk.QueryResultFlags.WAIT)
            compute_barrier(frame)
        period = vk.get_physical_device_properties_2(
            ctx.physical_device).properties.limits.timestamp_period * 1e-9
        best = [1e9] * len(cases)
        for _ in range(args.rounds):
            rec.submit()
            t = array.array('Q', bytes(stamps.map())[:8 * count])
            best = [min(b, (t[i + 1] - t[i]) * period) for i, b in enumerate(best)]
        vk.destroy_query_pool(ctx.device, pool)
        print(ctx.device_name)
        totals = {'f32': 0.0, 'bf16': 0.0}
        for (label, flops, _), t in zip(cases, best):
            totals[label.split()[-1]] += t
            print(f'  {label:18s} {t * 1000:8.3f} ms  {flops / t / 1e12:6.1f} TFLOP/s')
        print(f'  total f32 {totals["f32"] * 1000:.2f} ms, bf16 {totals["bf16"] * 1000:.2f} ms '
              f'({totals["f32"] / totals["bf16"]:.2f}x)')


if __name__ == '__main__':
    main()
