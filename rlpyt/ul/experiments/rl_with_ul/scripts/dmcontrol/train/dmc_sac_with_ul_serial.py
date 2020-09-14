
import sys
import pprint

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.ul.envs.dmcontrol import make
from rlpyt.ul.algos.rl_with_ul.sac_with_ul import SacWithUl
from rlpyt.ul.agents.dmc_sac_agent import SacAgent
from rlpyt.adam.envstep_runner import MinibatchRlEvalEnvStep
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.ul.experiments.rl_with_ul.configs.dmc_sac_with_ul import configs


def build_and_train(
        slot_affinity_code="0slt_1gpu_1cpu",
        log_dir="test",
        run_ID="0",
        config_key="sac_with_ul",
        ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    pprint.pprint(config)

    sampler = SerialSampler(
        EnvCls=make,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        # TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["env"],  # Same args!
        **config["sampler"]
    )
    algo = SacWithUl(**config["algo"])
    agent = SacAgent(
        conv_kwargs=config["conv"],
        fc1_kwargs=config["fc1"],
        pi_model_kwargs=config["pi_model"],
        q_model_kwargs=config["q_model"],
        **config["agent"])
    runner = MinibatchRlEvalEnvStep(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        frame_skip=config["env"]["frame_skip"],
        **config["runner"]
    )
    name = config["env"]["domain_name"] + "_" + config["env"]["task_name"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
