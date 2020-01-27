

from copy import deepcopy
from collections import namedtuple
import os.path as osp
import json

VARIANT = "variant_config.json"

VariantLevel = namedtuple("VariantLevel", ["keys", "values", "dir_names"])

VariantLevel.__doc__ += (
    "\nA `namedtuple` which describes a set of hyperparameter settings.  "
    "\n\nInput ``keys`` should be a list of tuples, where each tuple is the sequence "
    "of keys to navigate down the configuration dictionary to the value. "
    "\n\nInput ``values`` should be a list of lists, where each element of the outer "
    "list is a complete set of values, and position in the inner list "
    "corresponds to the key at that position in the keys list, i.e. each combination "
    "must be explicitly written. "
    "\n\nInput ``dir_names`` should have the same length as ``values``, and include"
    "unique paths for logging results from each value combination."
)


def make_variants(*variant_levels):
    """Takes in any number of ``VariantLevel`` objects and crosses them in order.
    Returns the resulting lists of full variant and log directories.  Every
    set of values in one level is paired with every set of values in the next
    level, e.g. if two combinations are specified in one level and three
    combinations in the next, then six total configuations will result.

    Use in the script to create and run a set of learning runs.
    """
    variants, log_dirs = [dict()], [""]
    for variant_level in variant_levels:
        variants, log_dirs = _cross_variants(variants, log_dirs, variant_level)
    return variants, log_dirs


def _cross_variants(prev_variants, prev_log_dirs, variant_level):
    """For every previous variant, make all combinations with new values."""
    keys, values, dir_names = variant_level
    assert len(prev_variants) == len(prev_log_dirs)
    assert len(values) == len(dir_names)
    assert len(keys) == len(values[0])
    assert all(len(values[0]) == len(v) for v in values)

    variants = list()
    log_dirs = list()
    for prev_variant, prev_log_dir in zip(prev_variants, prev_log_dirs):
        for vs, n in zip(values, dir_names):
            variant = deepcopy(prev_variant)
            log_dir = osp.join(prev_log_dir, n)
            if log_dir in log_dirs:
                raise ValueError("Names must be unique.")
            for v, key_path in zip(vs, keys):
                current = variant
                for k in key_path[:-1]:
                    if k not in current:
                        current[k] = dict()
                    current = current[k]
                current[key_path[-1]] = v
            variants.append(variant)
            log_dirs.append(log_dir)
    return variants, log_dirs


def load_variant(log_dir):
    """Loads the `variant.json` file from the directory."""
    with open(osp.join(log_dir, VARIANT), "r") as f:
        variant = json.load(f)
    return variant


def save_variant(variant, log_dir):
    """Saves a `variant.json` file to the directory."""
    with open(osp.join(log_dir, VARIANT), "w") as f:
        json.dump(variant, f)


def update_config(default, variant):
    """Performs deep update on all dict structures from ``variant``, updating only
    individual fields.  Any field in ``variant`` must be present in ``default``,
    else raises ``KeyError`` (helps prevent mistakes).  Operates recursively to
    return a new dictionary."""
    new = default.copy()
    for k, v in variant.items():
        if k not in new:
            raise KeyError(f"Variant key {k} not found in default config.")
        if isinstance(v, dict) != isinstance(new[k], dict):
            raise TypeError(f"Variant dict structure at key {k} mismatched with"
                " default.")
        new[k] = update_config(new[k], v) if isinstance(v, dict) else v
    return new
