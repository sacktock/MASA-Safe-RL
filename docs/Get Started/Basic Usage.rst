Basic Usage
===========

This page shows the **minimal** way to use MASA *without* :func:`masa.common.utils.make_env`, by manually
constructing a Gymnasium environment and wrapping it in the recommended order:

    :class:`~gymnasium.wrappers.TimeLimit` :math:`\rightarrow` 
    :class:`~masa.common.labelled_env.LabelledEnv` :math:`\rightarrow` 
    :class:`~masa.common.constraints.base.BaseConstraintEnv`  :math:`\rightarrow` 
    :class:`~masa.common.wrappers.ConstraintMonitor` :math:`\rightarrow` 
    :class:`~masa.common.wrappers.RewardMonitor`

This is the same order enforced by :func:`~masa.common.utils.make_env` (notably, ``TimeLimit`` must come first).

Overview
--------

MASA components reason over **labels** (atomic predicates) derived from observations. The wrapper
:class:`masa.common.labelled_env.LabelledEnv` computes these labels on every :meth:`gymnasium.Env.reset`
and :meth:`gymnasium.Env.step` and stores them in ``info["labels"]``.

Constraints then consume these labels and expose consistent metrics, while the monitor wrappers
attach step/episode summaries to the ``info`` dictionary for logging and debugging.

Minimal environment construction
--------------------------------

.. code-block:: python

    import gymnasium as gym

    # Core MASA wrappers
    from masa.common.wrappers import TimeLimit, ConstraintMonitor, RewardMonitor
    from masa.common.labelled_env import LabelledEnv

    # Simple Media Streaming environment
    from masa.env.tabular.media_streaming import MediaStreaming

    # Example constraint wrapper (a BaseConstraintEnv implementation)
    from masa.common.constraints.cmdp import CumulativeCostEnv

    # --- 1) Define label and cost functions ---

    def label_fn(obs):
        """
        Example labelling function for MediaStreaming-like observations.

        Returns:
            set[str]: Atomic predicates holding in the current observation.
        """
        labels = set()
        # These keys are illustrative; adapt to your observation structure.
        try:
            if int(obs) == 0:
                labels.add("unsafe")
        except:
            return set()
        return labels

    def cost_fn(labels):
        """
        Example 0/1 cost: unsafe if the 'unsafe' predicate holds.
        """
        return 1.0 if "unsafe" in labels else 0.0

    # --- 1.5) Or use default label_fn and cost_fn supplied by the environment (recommended)
    from masa.env.tabular.media_streaming import label_fn, cost_fn

    # --- 2) Build the environment and wrap in the correct order ---

    env = MediaStreaming()

    # Recommended: apply TimeLimit first (episode length is enforced before anything else).
    env = TimeLimit(env, max_episode_steps=1_000)

    # Attach labels to info["labels"] at every reset/step.
    env = LabelledEnv(env, label_fn)

    # Apply a constraint wrapper (example: cumulative cost/budget style constraint).
    # Typical kwargs are shown; consult the constraint's docstring / Constraints API reference.
    env = CumulativeConstraintEnv(env, cost_fn=cost_fn, budget=25.0)

    # Finally, attach monitoring wrappers for constraints and reward logging.
    env = ConstraintMonitor(env)
    env = RewardMonitor(env)

Random-agent interaction loop (Gymnasium-style)
-----------------------------------------------

.. code-block:: python

    import numpy as np

    num_episodes = 3

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)

        ep_return = 0.0
        ep_len = 0

        # Your monitors/constraints may attach additional fields; labels are always in info["labels"]
        labels = info.get("labels", set())
        print(f"[episode {ep}] reset labels={labels}")

        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1

            labels = info.get("labels", set())

            # Common pattern: monitors expose step metrics in info (names may vary by constraint).
            constraint = info.get("constraint", {})
            if isinstance(constraint, dict) and "step" in constraint:
                step_cost = constraint["step"].get("cost", 0.0)
                violated = constraint["step"].get("violated", False)

            if violated:
                print(f"  step={ep_len:04d} VIOLATION labels={labels} cost={step_cost}")

        # Episode-end metrics are often attached on the final transition by the monitors.
        # Again, keys vary; print what you care about.
        constraint = info.get("constraint", {})
        if isinstance(constraint, dict) and "episode" in constraint:
            ep_cost = constraint["episode"].get("cum_cost", None)
            ep_satisfied = constraint["episode"].get("satisfied", None)

        print(
            f"[episode {ep}] return={ep_return:.2f} len={ep_len} "
            f"episode_cost={ep_cost} episode_satisfied={ep_satisfied}"
       )

Training with PPO
-----------------

Below is a minimal example showing how to initialize and train PPO (provided by MASA) using the wrapped environment.
The specific PPO constructor and train API may include additional options (e.g., logging, eval env,
saving); the snippet mirrors the general style used in MASA runs.

.. code-block:: python

   from masa.algorithms.ppo import PPO

   # (Optional) create a separate evaluation environment with the same wrapper stack.
   def make_eval_env():
       eval_env = MediaStreaming()
       eval_env = TimeLimit(eval_env, max_episode_steps=1_000)
       eval_env = LabelledEnv(eval_env, label_fn)
       eval_env = CumulativeConstraintEnv(eval_env, cost_fn=cost_fn, budget=25.0)
       eval_env = ConstraintMonitor(eval_env)
       eval_env = RewardMonitor(eval_env)
       return eval_env

   eval_env = make_eval_env()

   # Initialize PPO.
   # Common kwargs (device, seed, logging) follow the same pattern as other MASA algorithms.
   algo = PPO(
       env,
       seed=0,
       device="auto",
       verbose=1,
       eval_env=eval_env,            # optional
       tensorboard_logdir=None,      # optional
   )

   # Train PPO. MASA algorithms automatically support eval/log frequencies and windowed stats.
   algo.train(
       total_timesteps=200_000,
       num_eval_episodes=10,         # optional
       eval_freq=10_000,             # optional
       log_freq=2_000,               # optional
       stats_window_size=100,        # optional
   )

API Reference for :func:`~masa.common.utils.make_env`
---------------------------------------

.. autofunction:: masa.common.utils.make_env

Next Steps
----------

- :doc:`Constraints API Reference <../Common/Constraints>` - View the common constraints supported by MASA.
