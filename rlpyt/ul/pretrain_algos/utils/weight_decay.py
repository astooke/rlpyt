"""
Took these functions from:
https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3

To add weight decay but not apply it to batch norm or biases.
"""


def add_weight_decay(model, weight_decay=0., filter_ndim_1=True,
        skip_list=None):
    """Returns parameters and weight_decay args for optimizer."""
    skip_list = [] if skip_list is None else skip_list
    if weight_decay == 0 or (not filter_ndim_1 and not skip_list):
        return model.parameters(), weight_decay
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (filter_ndim_1 and len(param.shape) == 1) or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    params = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
    ]
    weight_decay = 0.
    return params, weight_decay
