"""Time training steps and each kernel dispatch of one step.

    python -m vabs_gpt.bench --dim 128 --layers 5 --heads 8 --mlp 448 --batch 32

Prints ms/step for whole training steps on random bytes, then the GPU time of
every dispatch of a step (from timestamp queries), summed per kernel.
"""
import argparse
import array
import os
import time

import vk
from vabs.memory import MemoryUsage

from .gpu import ComputeContext, Recording, compute_barrier
from .model import Config, Device, Model, Muon, Runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctx', type=int, default=256)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--layers', type=int, default=5)
    ap.add_argument('--mlp', type=int, default=448)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=200)
    ap.add_argument('--no-profile', action='store_true')
    ap.add_argument('--optimizer', choices=('adamw', 'muon', 'muonclip'), default='adamw')
    ap.add_argument('--qk-clip', type=float, default=None, help='default: 100 with muonclip')
    ap.add_argument('--optim-bits', type=int, choices=(32, 8), default=32)
    args = ap.parse_args()
    cfg = Config(ctx=args.ctx, dim=args.dim, heads=args.heads, layers=args.layers, mlp=args.mlp)
    with ComputeContext() as ctx:
        dev = Device(ctx)
        calls = []
        real_run = dev.run

        def logging_run(frame, kernel, push, n, gy=1, barrier=True):
            calls.append((kernel, push, n, gy, barrier))
            real_run(frame, kernel, push, n, gy, barrier)

        dev.run = logging_run
        model = Model(dev, cfg, optim_bits=args.optim_bits, muon=args.optimizer != 'adamw')
        qk = args.qk_clip if args.qk_clip is not None else (
            100.0 if args.optimizer == 'muonclip' else 0.0)
        runner = Runner(model, args.batch, 'train',
                        muon=None if args.optimizer == 'adamw' else Muon(), qk_clip=qk)
        dev.run = real_run
        calls = calls[:len(calls) // len(runner.slots)]   # one recording; each slot has its own
        n = args.batch * cfg.ctx
        batches = [(os.urandom(n), os.urandom(n)) for _ in range(8)]
        for i in range(10):
            runner.train_step([batches[i % 8]], 1e-4, clip=1.0)
        def bursts(fn, per=10, count=40):
            times = []
            for _ in range(count):
                t0 = time.perf_counter()
                for i in range(per):
                    fn(i)
                times.append((time.perf_counter() - t0) / per * 1000)
            times.sort()
            return times[0], times[len(times) // 2]

        lo, mid = bursts(lambda i: runner.train_step([batches[i % 8]], 1e-4, clip=1.0))
        print(f'{ctx.device_name}: step min {lo:.2f} ms, median {mid:.2f} ms  '
              f'({n / lo * 1000:,.0f} tok/s at min)')
        lo, mid = bursts(lambda i: runner.submit_step([batches[i % 8]], 1e-4, clip=1.0))
        runner.collect()
        print(f'  queued a step ahead (as train does): min {lo:.2f} ms, median {mid:.2f} ms  '
              f'({n / lo * 1000:,.0f} tok/s at min)')
        if args.no_profile:
            return
        # One recording of the whole step with a GPU timestamp after every dispatch.
        # Each interval keeps its fastest sample over many submissions, which drops
        # the ones stretched by other programs using the GPU.
        # Dispatches recorded without a barrier may overlap, so they share one interval.
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
        for kernel, t in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f'  {kernel:34s} x{counts[kernel]:3d}  {t * 1000:7.3f} ms  {100 * t / total:5.1f}%')
        print(f'  {"sum":34s}       {total * 1000:7.3f} ms')


if __name__ == '__main__':
    main()
