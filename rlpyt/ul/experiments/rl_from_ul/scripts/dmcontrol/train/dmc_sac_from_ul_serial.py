
import sys
import pprint
import os.path as osp

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.ul.envs.dmcontrol import make
from rlpyt.ul.algos.rl_from_ul.rad_sac_from_ul import RadSacFromUl
from rlpyt.ul.agents.dmc_sac_agent import SacAgent
from rlpyt.adam.envstep_runner import MinibatchRlEvalEnvStep
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.ul.experiments.rl_from_ul.configs.dmc_sac_from_ul import configs


def build_and_train(
        slot_affinity_code="0slt_1gpu_1cpu",
        log_dir="test",
        run_ID="0",
        config_key="serial_radsac",
        experiment_title="exp",
        ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    # Hack that the first part of the log_dir matches the source of the model
    model_base_dir = config["pretrain"]["model_dir"]
    if model_base_dir is not None:
        raw_log_dir = log_dir.split(experiment_title)[-1].lstrip("/")  # get rid of ~/GitRepos/adam/rlpyt/data/local/<timestamp>/
        model_sub_dir = raw_log_dir.split("/RlFromUl/")[0]  # keep the UL part, which comes first
        pretrain_ID = config["pretrain"]["run_ID"]
        config["agent"]["state_dict_filename"] = osp.join(model_base_dir,
            model_sub_dir, f"run_{pretrain_ID}/params.pkl")

    pprint.pprint(config)

    sampler = SerialSampler(
        EnvCls=make,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        eval_env_kwargs=config["env"],  # Same args!
        **config["sampler"]
    )
    algo = RadSacFromUl(**config["algo"])
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
