"""Run: python -m masa.examples.preemptive_ltl_example"""
import numpy as np

from masa.common.constraints.ltl_safety import LTLSafetyEnv
from masa.common.labelled_env import LabelledEnv
from masa.deterministic_shield import PreemptiveLTLShield
from masa.envs.tabular.colour_bomb_grid_world import ColourBombGridWorld, label_fn
from masa.examples.colour_bomb_grid_world.property_2 import make_dfa


def main():
    # This existing DFA requires one extra step on a bomb before leaving it.
    base = ColourBombGridWorld(slip_prob=0.0)
    env = PreemptiveLTLShield(LTLSafetyEnv(
        LabelledEnv(base, label_fn), dfa=make_dfa(), obs_type="discrete"
    ))
    # Simple demonstration policy: prefer right, then stay. Replace these scores
    # with Q-values/logits. The mask is applied BEFORE choosing the action.
    scores = np.array([0.0, 10.0, 0.0, 0.0, 1.0])
    try:
        obs, info = env.reset(seed=0)
        for _ in range(20):
            safe = np.flatnonzero(info["action_mask"])
            action = int(safe[np.argmax(scores[safe])])
            old_state = obs % base.observation_space.n
            obs, _, terminated, truncated, info = env.step(action)
            assert not info["shield_intervened"]
            print(f"state={old_state}, safe={safe.tolist()}, chosen={action}, "
                  f"next={obs % base.observation_space.n}")
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
