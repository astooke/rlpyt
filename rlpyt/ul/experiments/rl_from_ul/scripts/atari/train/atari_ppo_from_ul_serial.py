
import sys
import pprint
import os.path as osp

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.ul.envs.atari import AtariEnv84
from rlpyt.algos.pg.ppo import PPO
# from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
# from rlpyt.ul.agents.atari_dqn_rl_from_ul_agent import AtariDqnRlFromUlAgent
from rlpyt.ul.agents.atari_pg_agent import AtariPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.ul.experiments.rl_from_ul.configs.atari_ppo_from_ul import configs


def build_and_train(
        slot_affinity_code="0slt_1gpu_1cpu",
        log_dir="test",
        run_ID="0",
        config_key="ppo_16env",
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
        config["agent"]["state_dict_filename"] = osp.join(model_base_dir,
            model_sub_dir, "run_0/params.pkl")
    pprint.pprint(config)

    sampler = SerialSampler(
        EnvCls=AtariEnv84,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["env"],  # Same args!
        **config["sampler"]
    )
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariPgAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
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
