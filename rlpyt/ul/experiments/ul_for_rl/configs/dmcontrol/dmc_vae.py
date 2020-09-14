
import copy

configs = dict()

config = dict(
    algo=dict(
        replay_filepath=None,
        batch_size=128,
        delta_T=0,
        learning_rate=1e-3,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=1000,  # number of updates
        clip_grad_norm=10.,
        latent_size=128,
        hidden_sizes=None,
        kl_coeff=1.0,
        onehot_action=False,
        activation_loss_coefficient=0.,  # rarely if ever use
        validation_split=0.0,
        n_validation_batches=0,  # usually don't do it.
    ),
    encoder=dict(
        channels=[32, 32, 32, 32],
        kernel_sizes=[3, 3, 3, 3],
        strides=[2, 2, 2, 1],
        paddings=None,
        hidden_sizes=None,
        kiaming_init=True,
    ),
    decoder=dict(
        reshape=(32, 9, 9),
        channels=(32, 32, 32, 9),
        kernel_sizes=(3, 3, 3, 3),
        strides=(2, 2, 2, 1),
        paddings=(0, 0, 0, 0),
        output_paddings=(0, 1, 1, 0),
    ),
    optim=dict(
        weight_decay=0,
    ),
    runner=dict(
        n_updates=int(2e4),
        log_interval_updates=int(1e3),
    ),
    name="dmc_vae",  # probably change this with the filepath
)

configs["dmc_vae"] = config
