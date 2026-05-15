from __future__ import annotations


def test_reach_avoid_make_env_tracks_avoid_and_reach_labels():
    from masa.common.utils import make_env
    from masa.envs.tabular.colour_grid_world import label_fn
    from masa.plugins.helpers import load_plugins

    load_plugins()

    def build_env():
        return make_env(
            "colour_grid_world",
            "reach_avoid",
            40,
            label_fn=label_fn,
            avoid_label="blue",
            reach_label="goal",
        )

    unsafe_env = build_env()
    obs, info = unsafe_env.reset(seed=1)
    assert obs == 0
    assert info["constraint"]["type"] == "reach_avoid"

    for action in [2, 2, 2, 2]:
        obs, reward, terminated, truncated, info = unsafe_env.step(action)

    assert obs == 36
    assert info["labels"] == {"blue"}
    assert info["constraint"]["step"]["cost"] == 1.0
    assert info["constraint"]["step"]["violation"] is True
    assert info["constraint"]["episode"]["violated"] is True
    assert info["constraint"]["episode"]["satisfied"] == 0.0
    assert terminated is False
    assert truncated is False
    unsafe_env.close()

    goal_env = build_env()
    goal_env.reset(seed=4)

    for action in [2] * 8 + [1] * 8:
        obs, reward, terminated, truncated, info = goal_env.step(action)
        if terminated or truncated:
            break

    assert obs == 80
    assert reward == 1.0
    assert info["labels"] == {"goal"}
    assert info["constraint"]["step"]["reached"] is True
    assert info["constraint"]["episode"]["reached"] is True
    assert info["constraint"]["episode"]["violated"] is False
    assert info["constraint"]["episode"]["satisfied"] == 1.0
    assert terminated is True
    assert truncated is False
    goal_env.close()
