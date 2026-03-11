from isaacgymenvs.tasks import make_task


def test_init_reset_step_static_bridge():
    env = make_task(
        "Go2Bridge",
        {
            "num_envs": 4,
            "num_observations": 8,
            "num_actions": 12,
            "max_episode_length": 5,
            "bridge_type": "narrow_static_bridge",
        },
    )
    obs = env.reset()
    assert len(obs) == 4
    assert len(obs[0]) == 8

    actions = [[0.0 for _ in range(12)] for _ in range(4)]
    result = env.step(actions)
    assert len(result.obs) == 4
    assert len(result.rewards) == 4
    assert len(result.dones) == 4
    assert result.infos["terrain"]["kind"] == "bridge"


def test_fixed_plane_option():
    env = make_task(
        "Go2Bridge",
        {
            "num_envs": 1,
            "num_observations": 8,
            "num_actions": 12,
            "bridge_type": "fixed_plane",
        },
    )
    env.reset()
    result = env.step([[0.0 for _ in range(12)]])
    assert result.infos["terrain"]["kind"] == "plane"
