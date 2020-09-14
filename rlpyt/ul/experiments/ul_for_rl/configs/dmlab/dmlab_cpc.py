
import copy

configs = dict()

config = dict(
    algo=dict(
        replay_filepath=None,
        batch_B=16,
        batch_T=32,
        warmup_T=16,
        learning_rate=1e-3,
        rnn_size=256,
        latent_size=256,
        clip_grad_norm=1000.,
        onehot_actions=True,
        activation_loss_coefficient=0.,  # 0 for OFF
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=1000,  # number of updates
        validation_split=0.0,
        n_validation_batches=0,
    ),
    encoder=dict(
        use_fourth_layer=True,
        skip_connections=True,
        hidden_sizes=None,
        kiaming_init=True,
    ),
    optim=dict(
        weight_decay=0,
    ),
    runner=dict(
        n_updates=int(4e4),  # 40k Usually sufficient for one level?
        log_interval_updates=int(1e3),
    ),
    name="dmlab_cpc",  # probably change this with the filepath
)

configs["dmlab_cpc"] = config
