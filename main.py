 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/strategy_optimizer.py b/strategy_optimizer.py
new file mode 100644
index 0000000000000000000000000000000000000000..7c040c0072d2429dbaf91d94ee42be0e9554607d
--- /dev/null
+++ b/strategy_optimizer.py
@@ -0,0 +1,549 @@
+#!/usr/bin/env python
+"""
+Custom RL-based search strategy for AutoCkt.
+
+This file is derived from the user-supplied inference script and adds
+heuristics to address the observed failure modes (insufficient gain and
+phase margin, excessive bias current). The changes focus on:
+
+1) Stronger error shaping toward hard constraints.
+2) Guided action noise that biases exploration toward fixing the largest
+   spec deficits while still leveraging the trained policy.
+3) A lightweight local hill-climb pass around the best-known incumbent to
+   escape plateaus the policy repeatedly returned to in the recorded run.
+
+The script keeps the original API surface for loading an RLlib checkpoint
+and evaluating the TwoStageAmp environment so it can be swapped in for the
+previous inference script.
+"""
+
+import argparse
+import json
+import os
+from typing import Dict, Iterable, List, Optional, Tuple
+
+import numpy as np
+import ray
+from ray.rllib.agents.registry import get_agent_class
+from ray.tune.registry import register_env
+
+from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp
+
+ALGO_NAME = "PPO"
+ENV_NAME = "opamp-v0"
+OUTPUT_CIR = "final_design_lin.cir"
+
+# Search budget. These match the user's previous settings, but can be tuned
+# from the CLI if needed.
+MAX_RESETS = 300
+STEPS_PER_EPISODE = 100
+
+register_env(ENV_NAME, lambda config: TwoStageAmp(config))
+
+
+class Goal:
+    """Spec goals used by the environment."""
+
+    def __init__(self, d: Dict[str, float]):
+        missing = [k for k in ("gain_min", "ibias_max", "phm_min", "ugbw_min") if k not in d]
+        if missing:
+            raise ValueError("Missing keys in spec json: {}".format(missing))
+        self.gain_min = float(d["gain_min"])
+        self.ibias_max = float(d["ibias_max"])
+        self.phm_min = float(d["phm_min"])
+        self.ugbw_min = float(d["ugbw_min"])
+
+    @staticmethod
+    def load(path: str) -> "Goal":
+        with open(path, "r", encoding="utf-8") as f:
+            return Goal(json.load(f))
+
+    def to_env_specs(self) -> np.ndarray:
+        d = {
+            "gain_min": self.gain_min,
+            "ibias_max": self.ibias_max,
+            "phm_min": self.phm_min,
+            "ugbw_min": self.ugbw_min,
+        }
+        return np.array([d[k] for k in sorted(d.keys())], dtype=float)
+
+    def __repr__(self) -> str:  # pragma: no cover - debugging helper
+        return (
+            "Goal(gain_min={}, ibias_max={}, phm_min={}, ugbw_min={})".format(
+                self.gain_min, self.ibias_max, self.phm_min, self.ugbw_min
+            )
+        )
+
+
+class Incumbent:
+    """Tracks the best design seen so far."""
+
+    def __init__(self) -> None:
+        self.error = float("inf")  # type: float
+        self.params = None  # type: Optional[np.ndarray]
+        self.meas = None  # type: Optional[np.ndarray]
+
+    def has_value(self) -> bool:
+        return self.params is not None and self.meas is not None
+
+    def is_done(self) -> bool:
+        return self.error == 0.0 and self.has_value()
+
+    def consider(self, error: float, params: np.ndarray, meas: np.ndarray) -> bool:
+        if not np.isfinite(error):
+            return False
+
+        if error < self.error:
+            self.error = float(error)
+            self.params = params.copy()
+            self.meas = meas.copy() if isinstance(meas, np.ndarray) else np.array(meas, dtype=float)
+            return True
+        return False
+
+
+# ---------------------------------------------------------------------------
+# Helper utilities
+# ---------------------------------------------------------------------------
+
+
+def _get_params(env: TwoStageAmp) -> np.ndarray:
+    return env.cur_params_idx
+
+
+def _set_params(env: TwoStageAmp, params: np.ndarray) -> None:
+    env.cur_params_idx = params
+
+
+def _spec_deficits(meas: Iterable[float], goal: Goal) -> Dict[str, float]:
+    gain, ibias, phm, ugbw = [float(m) for m in meas]
+    return {
+        "gain": max(0.0, goal.gain_min - gain),
+        "ibias": max(0.0, ibias - goal.ibias_max),
+        "phm": max(0.0, goal.phm_min - phm),
+        "ugbw": max(0.0, goal.ugbw_min - ugbw),
+    }
+
+
+def _error(meas: Iterable[float], goal: Goal) -> float:
+    if not all(np.isfinite(v) for v in meas):
+        return float("inf")
+
+    gain_def, ibias_def, phm_def, ugbw_def = _spec_deficits(meas, goal).values()
+
+    # Aggressive penalties on bias overflow and gain shortfall; quadratic loss
+    # ensures large violations dominate the ranking of candidates.
+    total = 0.0
+    total += (gain_def / 25.0) ** 2  # drive toward >= gain_min quickly
+    total += (ibias_def / 5e-4) ** 2  # bias is the most violated in logs
+    total += (phm_def / 10.0) ** 2
+    total += (ugbw_def / 2e6) ** 2
+    return float(total) if np.isfinite(total) else float("inf")
+
+
+def _all_done(meas: Iterable[float], goal: Goal) -> bool:
+    g, i, p, u = [float(m) for m in meas]
+    return g >= goal.gain_min and i <= goal.ibias_max and p >= goal.phm_min and u >= goal.ugbw_min
+
+
+def _format_meas(meas: Iterable[float]) -> str:  # pragma: no cover - logging helper
+    g, i, p, u = [float(m) for m in meas]
+    return "G={:.1f} I={:.2e} PM={:.1f} UGBW={:.2e}".format(g, i, p, u)
+
+
+# ---------------------------------------------------------------------------
+# RLlib checkpoint loading helpers
+# ---------------------------------------------------------------------------
+
+
+def _read_rllib_params(checkpoint_path: str) -> Dict:
+    config_dir = os.path.dirname(checkpoint_path)
+    probe = [
+        os.path.join(config_dir, "params.json"),
+        os.path.join(config_dir, "../params.json"),
+    ]
+
+    cfg = {}  # type: Dict
+    for p in probe:
+        if os.path.exists(p):
+            with open(p, "r", encoding="utf-8") as f:
+                cfg = json.load(f)
+            break
+
+    cfg.setdefault("model", {})
+    cfg["num_workers"] = 0
+    cfg["num_gpus"] = 0
+    cfg["model"]["fcnet_hiddens"] = [64, 64]
+    cfg["model"]["fcnet_activation"] = "tanh"
+    return cfg
+
+
+def _boot_ray() -> None:
+    tmp = os.environ.get("RAY_TMPDIR") or os.path.join(os.path.expanduser("~"), ".ray_tmp")
+    os.makedirs(tmp, exist_ok=True)
+    ray.init(temp_dir=tmp, ignore_reinit_error=True)
+
+
+def _restore_policy(checkpoint_path: str, cfg: Dict):
+    cls = get_agent_class(ALGO_NAME)
+    agent = cls(env=ENV_NAME, config=cfg)
+    agent.restore(checkpoint_path)
+    return agent
+
+
+# ---------------------------------------------------------------------------
+# Post-processing helpers
+# ---------------------------------------------------------------------------
+
+
+def _emit_cir(env: TwoStageAmp, params: np.ndarray, out_path: str) -> None:
+    vals = [env.params[i][int(params[i])] for i in range(len(env.params_id))]
+    d = dict(zip(env.params_id, vals))
+
+    def as_int(v, default=1):
+        try:
+            return int(v)
+        except Exception:
+            return default
+
+    mp1 = as_int(d.get("mp1", 1))
+    mn1 = as_int(d.get("mn1", 1))
+    mn3 = as_int(d.get("mn3", 1))
+    mp3 = as_int(d.get("mp3", 1))
+    mn4 = as_int(d.get("mn4", 1))
+    mn5 = as_int(d.get("mn5", 1))
+    cc = float(d.get("cc", 1e-12))
+
+    cir = """two_stage_amp test
+* Two stage OPAMP
+.include /tmp/ngspice_model/45nm_bulk.txt
+
+*********** GENERATED BY AUTOCKT ***********
+.param wp1=0.5u lp1=90n mp1={mp1}
+.param wn1=0.5u ln1=90n mn1={mn1}
+.param wn3=0.5u ln3=90n mn3={mn3}
+.param wp3=0.5u lp3=90n mp3={mp3}
+.param wn4=0.5u ln4=90n mn4={mn4}
+.param wn5=0.5u ln5=90n mn5={mn5}
+.param cc={cc}
+********************************************
+
+.param ibias=30u
+.param cload=10p
+.param vcm=0.6
+ 
+mp1 net4 net4 VDD VDD pmos w=wp1 l=lp1 m=mp1
+mp2 net5 net4 VDD VDD pmos w=wp1 l=lp1 m=mp1
+mn1 net4 net2 net3 net3 nmos w=wn1 l=ln1 m=mn1
+mn2 net5 net1 net3 net3 nmos w=wn1 l=ln1 m=mn1
+mn3 net3 net7 VSS VSS nmos w=wn3 l=ln3 m=mn3
+mn4 net7 net7 VSS VSS nmos w=wn4 l=ln4 m=mn4
+mp3 net6 net5 VDD VDD pmos w=wp3 l=lp3 m=mp3
+mn5 net6 net7 VSS VSS nmos w=wn5 l=ln5 m=mn5
+cc net5 net6 cc
+ibias VDD net7 ibias
+
+vin in 0 dc=0 ac=1.0
+ein1 net1 cm in 0 0.5
+ein2 net2 cm in 0 -0.5
+vcm cm 0 dc=vcm
+
+vdd VDD 0 dc=1.2
+vss 0 VSS dc=0
+CL net6 0 cload
+
+.end
+""".format(
+        mp1=mp1,
+        mn1=mn1,
+        mn3=mn3,
+        mp3=mp3,
+        mn4=mn4,
+        mn5=mn5,
+        cc="{:.6e}".format(cc),
+    )
+
+    with open(out_path, "w", encoding="utf-8") as f:
+        f.write(cir)
+
+
+# ---------------------------------------------------------------------------
+# Strategy utilities
+# ---------------------------------------------------------------------------
+
+
+def _tune_comp_cap(env: TwoStageAmp, goal: Goal, inc: Incumbent, tries: int) -> None:
+    if (not inc.has_value()) or inc.is_done():
+        return
+
+    if "cc" not in env.params_id:
+        return
+
+    g_done = float(inc.meas[0]) >= goal.gain_min
+    i_done = float(inc.meas[1]) <= goal.ibias_max
+    if (not g_done) or (not i_done):
+        return
+
+    cc_pos = env.params_id.index("cc")
+    params = inc.params.copy()
+    noop = np.full(len(env.params_id), 1, dtype=int)
+
+    for _ in range(int(tries)):
+        _set_params(env, params.copy())
+        env.step(noop)
+        meas = env.cur_specs
+
+        err = 0.0 if _all_done(meas, goal) else _error(meas, goal)
+        inc.consider(err, params, meas)
+
+        if err == 0.0:
+            return
+
+        pm_def = _spec_deficits(meas, goal)["phm"]
+        u_def = _spec_deficits(meas, goal)["ugbw"]
+
+        if pm_def > 0 and params[cc_pos] < len(env.params[cc_pos]) - 1:
+            params[cc_pos] += 1
+        elif u_def > 0 and params[cc_pos] > 0:
+            params[cc_pos] -= 1
+        else:
+            return
+
+
+def _spec_guided_local_search(env: TwoStageAmp, goal: Goal, inc: Incumbent, budget: int = 24) -> None:
+    """Iterative best-improvement search around the incumbent.
+
+    The earlier single-depth hill climb was not strong enough to correct the
+    observed gain and bias violations. This variant repeatedly evaluates all
+    single-step neighbors, accepts the best improvement, and stops when no
+    neighbor helps. The small budget bounds runtime while still letting us
+    descend several steps if a clear path exists.
+    """
+
+    if not inc.has_value():
+        return
+
+    noop = np.full(len(env.params_id), 1, dtype=int)
+    params = inc.params.copy()
+    best_err = inc.error
+
+    for _ in range(int(budget)):
+        best_candidate = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]
+
+        for idx in range(len(params)):
+            for delta in (-2, -1, 1, 2):
+                candidate = params.copy()
+                candidate[idx] = np.clip(candidate[idx] + delta, 0, len(env.params[idx]) - 1)
+                _set_params(env, candidate)
+                env.step(noop)
+                meas = env.cur_specs
+                err = 0.0 if _all_done(meas, goal) else _error(meas, goal)
+                if err < best_err:
+                    best_err = err
+                    best_candidate = (candidate.copy(), meas.copy())
+
+        if best_candidate is None:
+            return
+
+        params, meas = best_candidate
+        inc.consider(best_err, params, meas)
+        if inc.is_done():
+            return
+
+
+def _guided_noise(deficits: Dict[str, float], size: Tuple[int, ...]) -> np.ndarray:
+    """Bias exploration toward the most violated specs.
+
+    Gain/phase deficits push actions upward (toward larger devices), while a
+    bias-current violation pushes actions downward to trim current.
+    """
+
+    # Base exploration level grows with aggregate deficit severity to improve
+    # escape probability when we are far from satisfying specs.
+    severity = (
+        deficits["gain"] / 150.0
+        + deficits["phm"] / 20.0
+        + deficits["ugbw"] / 8e6
+        + deficits["ibias"] / 3e-4
+    )
+    base_sigma = 0.6 + 0.3 * np.tanh(severity)
+
+    # Compute signed nudges. The mapping is coarse but captures the trends:
+    # - Positive gain/phase deficits → increase indices.
+    # - Positive bias deficit → decrease indices, weighted a bit higher to fight
+    #   the bias overruns seen in the previous run logs.
+    push_up = deficits["gain"] / 150.0 + deficits["phm"] / 25.0
+    push_down = 1.4 * deficits["ibias"] / 3e-4
+    direction = np.tanh(push_up - push_down)
+
+    # Center the noise around the bias direction so the rounding step still
+    # yields higher probability of moving along the signed preference.
+    return np.random.normal(direction * 0.8, base_sigma, size=size)
+
+
+# ---------------------------------------------------------------------------
+# Main search loop
+# ---------------------------------------------------------------------------
+
+
+def _search_strategy(agent, env: TwoStageAmp, goal: Goal) -> Incumbent:
+    inc = Incumbent()
+
+    try:
+        param_max = np.array([len(p) - 1 for p in env.params], dtype=int)
+    except Exception:
+        try:
+            # Fallback to gym MultiDiscrete style if env.params is unavailable
+            param_max = np.array(env.action_space.nvec - 1, dtype=int)
+        except Exception:
+            # Absolute fallback to a small action space; better than over-clipping
+            max_action_idx = len(getattr(env, "action_meaning", [0, 1, 2])) - 1
+            param_max = np.full(1, max_action_idx, dtype=int)
+
+    print(
+        "Starting Inference: Max Resets={}, Steps/Reset={}".format(
+            MAX_RESETS, STEPS_PER_EPISODE
+        )
+    )
+
+    for restart_idx in range(MAX_RESETS):
+        state = env.reset()
+        env.specs_ideal = goal.to_env_specs()
+
+        # Prime the incumbent with the reset point if it yields finite specs.
+        try:
+            current_meas = env.cur_specs
+            current_err = _error(current_meas, goal)
+            if np.isfinite(current_err):
+                inc.consider(current_err, _get_params(env), current_meas)
+        except Exception:
+            pass
+
+        step_count = 0
+        steps_since_improve = 0
+        done = False
+
+        while step_count < STEPS_PER_EPISODE and not done:
+            action = np.array(agent.compute_action(state))
+
+            # Bias the action toward fixing the most violated spec.
+            meas = env.cur_specs if hasattr(env, "cur_specs") else [0, 0, 0, 0]
+            deficits = _spec_deficits(meas, goal)
+            action = action + _guided_noise(deficits, action.shape)
+            action = np.rint(action).astype(int)
+            action = np.clip(action, 0, param_max)
+
+            state, reward, done, info = env.step(action)
+            meas = env.cur_specs
+
+            err = _error(meas, goal)
+            if not np.isfinite(err):
+                step_count += 1
+                continue
+            is_better = inc.consider(err, _get_params(env), meas)
+
+            if is_better and inc.error < 10.0:
+                print(
+                    "Reset {} | Step {} | New Best Err={:.4f} | {}".format(
+                        restart_idx + 1, step_count, inc.error, _format_meas(meas)
+                    )
+                )
+                steps_since_improve = 0
+            else:
+                steps_since_improve += 1
+
+            if err == 0.0:
+                print(
+                    "\n[SUCCESS] Solution found at Reset {}, Step {}".format(
+                        restart_idx + 1, step_count
+                    )
+                )
+                return inc
+
+            step_count += 1
+
+            # If we have not improved for a while, kick in a targeted jolt that
+            # pushes in the direction of the most violated spec. This helps
+            # escape flat regions where the policy oscillates.
+            if steps_since_improve >= 20 and step_count < STEPS_PER_EPISODE:
+                steps_since_improve = 0
+                # Bias toward fixing the largest deficit: gain/phase/ugbw push up,
+                # bias-current pushes down.
+                dominant = max(deficits, key=deficits.get)
+                directional = np.ones_like(action)
+                if dominant == "ibias":
+                    directional *= -1
+                nudged = action + directional * 2
+                nudged = np.clip(nudged, 0, param_max)
+                state, reward, done, info = env.step(nudged)
+                meas = env.cur_specs
+                err = _error(meas, goal)
+                if np.isfinite(err):
+                    inc.consider(err, _get_params(env), meas)
+
+        # Post-episode local search and CC tuning
+        _spec_guided_local_search(env, goal, inc, budget=24)
+        _tune_comp_cap(env, goal, inc, tries=50)
+
+        if inc.is_done():
+            print(
+                "\n[SUCCESS] Solution found after tuning at Reset {}".format(
+                    restart_idx + 1
+                )
+            )
+            return inc
+
+        if (restart_idx + 1) % 10 == 0:
+            perf = _format_meas(inc.meas) if inc.has_value() else "N/A"
+            print(
+                "Finished Reset {}/{}. Current Global Best Err: {:.4f} | {}".format(
+                    restart_idx + 1, MAX_RESETS, inc.error, perf
+                )
+            )
+
+    return inc
+
+
+# ---------------------------------------------------------------------------
+# CLI
+# ---------------------------------------------------------------------------
+
+
+def _cli() -> argparse.Namespace:
+    ap = argparse.ArgumentParser(description="AutoCkt inference (Guided Strategy)")
+    ap.add_argument("--model", required=True, type=str, help="Path to RLlib checkpoint")
+    ap.add_argument("--spec", required=True, type=str, help="Spec JSON")
+    return ap.parse_args()
+
+
+def main() -> None:
+    args = _cli()
+    goal = Goal.load(args.spec)
+    print("Goal: {}".format(goal))
+
+    cfg = _read_rllib_params(args.model)
+    _boot_ray()
+    agent = _restore_policy(args.model, cfg)
+
+    env = TwoStageAmp(
+        env_config={
+            "generalize": False,
+            "num_valid": 100,
+            "save_specs": False,
+            "run_valid": False,
+        }
+    )
+
+    inc = _search_strategy(agent, env, goal)
+
+    if not inc.has_value():
+        print("\n[FAIL] No candidate produced; nothing written")
+        return
+
+    _emit_cir(env, inc.params, OUTPUT_CIR)
+    print("\n[WRITE] {}".format(OUTPUT_CIR))
+    print("[FINAL BEST] err={:.4f} | {}".format(inc.error, _format_meas(inc.meas)))
+    print("params = {}".format(inc.params))
+
+
+if __name__ == "__main__":
+    main()
 
EOF
)
