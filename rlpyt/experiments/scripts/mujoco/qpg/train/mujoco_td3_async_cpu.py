
import sys

from rlpyt.utils.launching.affinity import affinity_from_code
# from rlpyt.samplers.serial_sampler import SerialSampler
from rlpyt.samplers.async_.async_cpu_sampler import AsyncCpuSampler
# from rlpyt.samplers.cpu.collectors import ResetCollector
from rlpyt.samplers.async_.collectors import DbCpuResetCollector
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.td3 import TD3
from rlpyt.agents.qpg.td3_agent import Td3Agent
# from rlpyt.runners.minibatch_rl_eval import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.experiments.configs.mujoco.qpg.mujoco_td3 import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    sampler = AsyncCpuSampler(
        EnvCls=gym_make,
        env_kwargs=config["env"],
        CollectorCls=DbCpuResetCollector,
        eval_env_kwargs=config["env"],
        **config["sampler"]
    )
    algo = TD3(optim_kwargs=config["optim"], **config["algo"])
    agent = Td3Agent(**config["agent"])
    runner = AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "async_td3_" + config["env"]["id"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
