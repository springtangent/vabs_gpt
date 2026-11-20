"""Time vabs and PyTorch training steps side by side on the same GPU.

    $env:PYTHONPATH = "vabs\\src"
    .ptvenv\\Scripts\\python.exe -m vabs_gpt.compare --dim 128 --layers 5 --heads 8 --mlp 448

Both models train on the same random bytes. The two take turns in short bursts,
so anything else using the GPU disturbs both alike. A burst is timed until the
loss of its last step is back on the host, and reported per step. Both run as
their training loops do: PyTorch reads each loss as it goes, vabs queues one
step ahead.
"""
import argparse
import os
import time

import torch

from .gpu import ComputeContext
from .model import Config, Device, Model, Runner, init_params
from .torch_ref import GPT, to_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctx', type=int, default=256)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--layers', type=int, default=5)
    ap.add_argument('--mlp', type=int, default=448)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--burst', type=int, default=20, help='steps per turn')
    ap.add_argument('--rounds', type=int, default=30)
    args = ap.parse_args()
    cfg = Config(ctx=args.ctx, dim=args.dim, heads=args.heads, layers=args.layers, mlp=args.mlp)
    n = args.batch * cfg.ctx
    data = [(os.urandom(n), os.urandom(n)) for _ in range(8)]
    device = torch.device('cuda')
    net = GPT(cfg).to(device)
    net.load_flat(init_params(cfg, 0))
    opt = torch.optim.AdamW(net.decay_groups(0.0), lr=1e-4, betas=(0.9, 0.99))

    def torch_step(i):
        inp, tgt = (to_tokens(x, args.batch, device) for x in data[i % 8])
        loss = net(inp, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        return loss.item()

    with ComputeContext() as ctx:
        runner = Runner(Model(Device(ctx), cfg), args.batch, 'train')

        def vabs_step(i):   # queued a step ahead, as the training loop does
            return runner.submit_step([data[i % 8]], 1e-4, beta2=0.99, clip=1.0)

        times = {'vabs': [], 'torch': []}
        for name, step in (('vabs', vabs_step), ('torch', torch_step)):
            for i in range(20):
                step(i)
        runner.collect()
        for _ in range(args.rounds):
            for name, step in (('vabs', vabs_step), ('torch', torch_step)):
                t0 = time.perf_counter()
                for i in range(args.burst):
                    step(i)
                runner.collect()   # the last queued vabs step (nothing for torch)
                times[name].append((time.perf_counter() - t0) / args.burst)
        print(f'{ctx.device_name}, {n:,} tokens/step')
        for name, ts in times.items():
            ts.sort()
            lo, mid = ts[0] * 1000, ts[len(ts) // 2] * 1000
            print(f'  {name:5s} min {lo:6.2f} ms  median {mid:6.2f} ms  '
                  f'({n / mid * 1000:,.0f} tok/s at median)')
        v, t = (times[k][len(times[k]) // 2] for k in ('vabs', 'torch'))
        print(f'  vabs / torch step time: {v / t:.2f}x at median, '
              f'{times["vabs"][0] / times["torch"][0]:.2f}x at min')


if __name__ == '__main__':
    main()
