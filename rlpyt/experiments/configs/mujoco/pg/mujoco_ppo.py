
import copy

configs = dict()

config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        clip_grad_norm=1e6,
        entropy_loss_coeff=0.0,
        gae_lambda=0.95,
        minibatches=32,
        epochs=10,
        ratio_clip=0.2,
        normalize_advantage=True,
        linear_lr_schedule=True,
    ),
    env=dict(id="Hopper-v3"),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=2048 * 10,
    ),
    sampler=dict(
        batch_T=2048,
        batch_B=1,
        max_decorrelation_steps=0,
    ),
)

configs["ppo_1M_serial"] = config

config = copy.deepcopy(configs["ppo_1M_serial"])

config = copy.deepcopy(configs["ppo_1M_serial"])
config["sampler"]["batch_B"] = 8
config["sampler"]["batch_T"] = 256
configs["ppo_1M_cpu"] = config

config["algo"]["minibatches"] = 1
config["algo"]["epochs"] = 32
configs["ppo_32ep_1mb"] = config
