
import sys
import pprint

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.ul.envs.atari import AtariEnv84
from rlpyt.ul.algos.rl_with_ul.dqn_with_ul import DqnUl
from rlpyt.ul.agents.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.ul.experiments.rl_with_ul.configs.atari_dqn_ul import configs


def build_and_train(
        slot_affinity_code="0slt_1gpu_1cpu",
        log_dir="test",
        run_ID="0",
        config_key="scaled_ddqn_ul",
        ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    pprint.pprint(config)

    sampler = SerialSampler(
        EnvCls=AtariEnv84,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["env"],  # Same args!
        **config["sampler"]
    )
    algo = DqnUl(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariDqnAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["game"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
