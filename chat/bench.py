"""Profile one training step of the chat model: GPU time per kernel.

    python -m vabs_gpt.chat.bench [--precision bf16] [--batch 8] [--dim 512 ...]

Records a step with a timestamp after every barrier, keeps each interval's
fastest sample over ``--rounds`` submissions (which drops the ones stretched by
other programs using the GPU) and sums them per kernel. Dispatches recorded
without a barrier between them share one interval.
"""
import argparse
import array
import os

import vk
from vabs.memory import MemoryUsage

from vabs_gpt.gpu import ComputeContext, Recording, compute_barrier
from vabs_gpt.model import Muon

from .model import PRECISIONS, ChatConfig, ChatModel, ChatRunner, Device


def main():
    ap = argparse.ArgumentParser()
    d = ChatConfig()
    for name in ('vocab', 'ctx', 'dim', 'heads', 'kv_heads', 'layers', 'mlp'):
        ap.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(d, name))
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--precision', choices=PRECISIONS, default='f32')
    ap.add_argument('--optimizer', choices=('adamw', 'muon'), default='muon')
    ap.add_argument('--rounds', type=int, default=20)
    args = ap.parse_args()
    cfg = ChatConfig(vocab=args.vocab, ctx=args.ctx, dim=args.dim, heads=args.heads,
                     kv_heads=args.kv_heads, layers=args.layers, mlp=args.mlp)
    with ComputeContext() as ctx:
        dev = Device(ctx, precision=args.precision)
        calls = []
        real_run = dev.run

        def logging_run(frame, kernel, push, n, gy=1, barrier=True):
            calls.append((kernel, push, n, gy, barrier))
            real_run(frame, kernel, push, n, gy, barrier)

        muon = args.optimizer == 'muon'
        model = ChatModel(dev, cfg, muon=muon)
        dev.run = logging_run
        runner = ChatRunner(model, args.batch, 'train', muon=Muon() if muon else None)
        dev.run = real_run
        calls = calls[:len(calls) // len(runner.slots)]   # one recording; each slot has its own
        n = args.batch * cfg.ctx
        tok = array.array('I', (int.from_bytes(os.urandom(2), 'little') % cfg.vocab
                                for _ in range(n)))
        runner.train_step([(tok, tok)], 1e-4)
        groups = [[]]
        for call in calls:
            groups[-1].append(call)
            if call[4]:
                groups.append([])
        groups = [g for g in groups if g]
        count = len(groups) + 1
        pool = vk.create_query_pool(ctx.device, vk.QueryPoolCreateInfo(
            query_type=vk.QueryType.TIMESTAMP, query_count=count))
        stamps, = dev.arena([8 * count], MemoryUsage.HOST_VISIBLE)
        rec = Recording(ctx)
        with rec.record() as frame:
            cmd = rec._cmd
            vk.cmd_reset_query_pool(cmd, pool, 0, count)
            vk.cmd_write_timestamp2(cmd, vk.PipelineStageFlags2.ALL_COMMANDS, pool, 0)
            for i, group in enumerate(groups):
                for call in group:
                    real_run(frame, *call)
                vk.cmd_write_timestamp2(cmd, vk.PipelineStageFlags2.ALL_COMMANDS, pool, i + 1)
            vk.cmd_copy_query_pool_results(cmd, pool, 0, count, stamps.arena.buffer_handle,
                                           stamps.offset, 8, vk.QueryResultFlags.RESULT_64
                                           | vk.QueryResultFlags.WAIT)
            compute_barrier(frame)
        period = vk.get_physical_device_properties_2(
            ctx.physical_device).properties.limits.timestamp_period * 1e-9
        best = [1e9] * len(groups)
        for _ in range(args.rounds):
            rec.submit()
            t = array.array('Q', bytes(stamps.map())[:8 * count])
            best = [min(b, (t[i + 1] - t[i]) * period) for i, b in enumerate(best)]
        vk.destroy_query_pool(ctx.device, pool)
        totals, counts = {}, {}
        for group, t in zip(groups, best):
            name = ' + '.join(dict.fromkeys(call[0] for call in group))
            totals[name] = totals.get(name, 0.0) + t
            counts[name] = counts.get(name, 0) + 1
        total = sum(totals.values())
        print(f'{ctx.device_name}, {args.precision}, {n} tokens per step:')
        for kernel, t in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f'  {kernel:40s} x{counts[kernel]:3d}  {t * 1000:7.3f} ms  {100 * t / total:5.1f}%')
        print(f'  {"sum":40s}       {total * 1000:7.3f} ms  ({n / total:,.0f} tok/s)')


if __name__ == '__main__':
    main()
