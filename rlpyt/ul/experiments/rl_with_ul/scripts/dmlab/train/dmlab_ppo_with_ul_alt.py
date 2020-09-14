
import sys
import pprint

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.ul.envs.dmlab import DmlabEnv
from rlpyt.ul.algos.rl_with_ul.ppo_with_ul import PpoUl
from rlpyt.ul.agents.dmlab_pg_agent import DmlabPgLstmAlternatingAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.ul.experiments.rl_with_ul.configs.dmlab_ppo_with_ul import configs


def build_and_train(
        slot_affinity_code="0slt_1gpu_1cpu",
        log_dir="test",
        run_ID="0",
        config_key="ppo_ul_16env",
        snapshot_mode="none", 
        snapshot_gap=None,
        ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    pprint.pprint(config)

    sampler = AlternatingSampler(
        EnvCls=DmlabEnv,
        env_kwargs=config["env"],
        CollectorCls=GpuWaitResetCollector,
        **config["sampler"]
    )
    algo = PpoUl(optim_kwargs=config["optim"], **config["algo"])
    agent = DmlabPgLstmAlternatingAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["level"]
    if snapshot_gap is not None:
        snapshot_gap = int(snapshot_gap)
    with logger_context(log_dir, run_ID, name, config,
            snapshot_mode=snapshot_mode, snapshot_gap=snapshot_gap):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
