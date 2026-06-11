"""WP4: SAC baseline training (5 seeds, budget logged). Run one seed per
process: python wp4_rl_train.py --seed N --steps 1000000"""
import sys, time, csv, json, argparse, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MOD = ROOT / "passport" / "models"
LOG = ROOT / "passport" / "logs"


def main(seed, steps):
    import torch
    torch.set_num_threads(3)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from ddwmr.rl_env import DDWMRPointEnv

    env = DDWMRPointEnv(seed=seed)

    class CurveCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.ep_r, self.ep_n = 0.0, 0

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.rows.append([self.num_timesteps, info["episode"]["r"],
                                      info["episode"]["l"]])
            return True

    cb = CurveCB()
    model = SAC("MlpPolicy", env, seed=seed, verbose=0,
                learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                train_freq=2, gradient_steps=1, gamma=0.995, tau=0.005,
                policy_kwargs=dict(net_arch=[256, 256]))
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=cb, log_interval=int(1e9))
    wall = time.perf_counter() - t0
    model.save(str(MOD / f"sac_seed{seed}"))
    with open(LOG / f"wp4_sac_curve_seed{seed}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["timesteps", "ep_return", "ep_len"])
        w.writerows(cb.rows)
    with open(LOG / f"wp4_sac_budget_seed{seed}.json", "w") as f:
        json.dump(dict(seed=seed, steps=steps, wall_s=wall,
                       fps=steps / wall), f)
    print(f"SAC seed {seed}: {steps} steps in {wall:.0f}s ({steps/wall:.0f} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=500_000)
    a = ap.parse_args()
    main(a.seed, a.steps)
