from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


def test_rlpyt_simple():
    """ partially copied from example 1 """
    game = "pong"
    run_ID = 0
    cuda_idx = None
    n_steps = 1
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DQN(min_steps_learn=1e3,
               replay_size=1e3)  # remove memory issues
    agent = AtariDqnAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=n_steps,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "test_example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()
