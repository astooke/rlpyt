
import sys
import pprint

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.adam.dmcontrol import make
from rlpyt.ul.algos.sac_ul import SacUl
from rlpyt.ul.agents.dmc_sac_with_ul_agent import SacWithUlAgent
from rlpyt.adam.envstep_runner import MinibatchRlEvalEnvStep
from rlpyt.utils.logging.context import logger_context
# from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.ul.experiments.configs.dmc_sac_with_ul import configs


def build_and_train(
        slot_affinity_code="0slt_0gpu_4cpu_4cpr",
        log_dir="test",
        run_ID="0",
        config_key="sac_ul_compress",
        ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    # variant = load_variant(log_dir)
    # config = update_config(config, variant)

    config["algo"]["min_steps_rl"] = 100
    config["algo"]["min_steps_ul"] = 150
    config["algo"]["replay_size"] = 1e4
    config["algo"]["batch_size"] = 64
    config["algo"]["ul_batch_size"] = 32
    config["runner"]["n_steps"] = 1e3
    config["runner"]["log_interval_steps"] = 1e2
    config["sampler"]["eval_n_envs"] = 1
    config["sampler"]["eval_max_steps"] = 500
    config["algo"]["stop_rl_conv_grad"] = True
    config["algo"]["ul_update_schedule"] = "cosine_8"

    pprint.pprint(config)

    sampler = SerialSampler(
        EnvCls=make,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        # TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["env"],  # Same args!
        **config["sampler"]
    )
    algo = SacUl(**config["algo"])
    agent = SacWithUlAgent(
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
