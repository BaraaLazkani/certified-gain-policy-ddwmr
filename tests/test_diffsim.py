"""Cross-validation: differentiable exact-ZOH rollout vs numpy RK4 eval sim."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from ddwmr.diffsim import rollout_metric, discretize
from ddwmr.sim import run_closedloop, LyapGains
from ddwmr.metrics import anchor_metric


def main():
    torch.set_num_threads(8)
    cases = [(0.6, 0.4, 0.8, 1.7), (0.6, 0.4, 0.5, 2.5),
             (-0.2, -0.5, 0.8, 1.7), (0.5, -0.5, 0.3, 1.0),
             (0.9, 0.3, 1.0, 3.0), (0.2, 0.1, 0.15, 0.6)]
    print(f"{'case':<28}{'e_l diff':>10}{'t diff':>9}{'th diff':>9}{'E diff %':>10}")
    for tx, ty, kp, kth in cases:
        tg = torch.tensor([[tx, ty]])
        g = torch.tensor([[kp, kth]])
        E, p = rollout_metric(g, tg, torch.tensor([1.0]), T=30.0, return_parts=True)
        r = run_closedloop(np.array([[tx, ty]]), "lyap",
                           LyapGains(Kp=np.array([kp]), Kth=np.array([kth])), T_max=30.0)
        m = anchor_metric(r.e_l_mean, r.theta_end, r.t_end, [[0, 0]], [[tx, ty]], C_l=1.0)
        dE = 100 * abs(E.item() - m["e_total"][0]) / max(m["e_total"][0], 1e-9)
        print(f"({tx},{ty}) Kp={kp} Kth={kth}: "
              f"{abs(p['e_l'].item() - r.e_l_mean[0])*1000:>8.3f}mm"
              f"{abs(p['t_soft'].item() - r.t_end[0]):>8.3f}s"
              f"{np.degrees(abs(p['theta_end'].item() - r.theta_end[0])):>8.3f}d"
              f"{dE:>9.2f}%")

    # gradient + speed benchmark at training scale
    B = 200
    rng = np.random.default_rng(0)
    ang = rng.uniform(-0.7 * np.pi, 0.7 * np.pi, B)
    rad = rng.uniform(0.1, 1.0, B)
    tg = torch.tensor(np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), dtype=torch.float32)
    g = torch.tensor(rng.uniform([0.1, 0.5], [1.0, 3.0], (B, 2)), dtype=torch.float32,
                     requires_grad=True)
    t0 = time.perf_counter()
    E = rollout_metric(g, tg, torch.ones(B), T=12.0)
    tf = time.perf_counter() - t0
    t0 = time.perf_counter()
    E.mean().backward()
    tb = time.perf_counter() - t0
    print(f"\nB={B} T=12s: fwd {tf:.1f}s bwd {tb:.1f}s | grad finite: "
          f"{np.isfinite(g.grad.numpy()).all()}")


if __name__ == "__main__":
    main()
