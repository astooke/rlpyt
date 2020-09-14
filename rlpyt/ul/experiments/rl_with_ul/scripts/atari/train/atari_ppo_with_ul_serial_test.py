
import sys
import pprint

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.adam.atari_env import AtariEnv84
# from rlpyt.algos.pg.ppo import PPO
from rlpyt.ul.algos.ppo_ul import PpoUl
# from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
# from rlpyt.ul.agents.atari_dqn_rl_from_ul_agent import AtariDqnRlFromUlAgent
# from rlpyt.ul.agents.atari_pg_rl_from_ul_agent import AtariPgRlFromUlAgent
from rlpyt.ul.agents.atari_pg_rl_with_ul_agent import AtariPgRlWithUlAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
# from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.ul.experiments.configs.atari_ppo_ul import configs


def build_and_train(
        slot_affinity_code="0slt_0gpu_4cpu_4cpr",
        log_dir="test",
        run_ID="0",
        config_key="ppo_ul_16env"
        ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    # variant = load_variant(log_dir)
    # config = update_config(config, variant)


    # config["sampler"]["batch_B"] = 4
    # config["sampler"]["batch_T"] = 5
    # config["runner"]["log_interval_steps"] = 100
    # config["runner"]["n_steps"] = 1000
    config["algo"]["ul_update_schedule"] = "constant_1"
    config["algo"]["min_steps_rl"] = 1e3
    config["algo"]["min_steps_ul"] = 200
    config["algo"]["max_steps_ul"] = 20e6
    config["model"]["stop_conv_grad"] = True
    config["sampler"]["max_decorrelation_steps"] = 0
    config["sampler"]["batch_B"] = 3
    config["sampler"]["batch_T"] = 20
    config["algo"]["ul_pri_alpha"] = 1.
    config["algo"]["ul_pri_n_step_return"] = 10
    config["algo"]["ul_replay_size"] = 900


    pprint.pprint(config)

    sampler = SerialSampler(
        EnvCls=AtariEnv84,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["env"],  # Same args!
        **config["sampler"]
    )
    algo = PpoUl(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariPgRlWithUlAgent(model_kwargs=config["model"], **config["agent"])
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
