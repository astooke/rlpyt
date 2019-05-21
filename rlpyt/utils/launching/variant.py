

from copy import deepcopy
from collections import namedtuple
import os.path as osp
import json

VARIANT = "variant_config.json"

VariantLevel = namedtuple("VariantLevel", ["keys", "values", "dir_names"])


def make_variants(*variant_levels):
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
    with open(osp.join(log_dir, VARIANT), "r") as f:
        variant = json.load(f)
    return variant


def save_variant(variant, log_dir):
    with open(osp.join(log_dir, VARIANT), "w") as f:
        json.dump(variant, f)


def update_config(default, variant):
    """Performs deep update on all dict structures from variant, updating only
    individual fields, which must be present in default."""
    new = default.copy()
    for k, v in variant.items():
        if k not in new:
            raise KeyError(f"Variant key {k} not found in default config.")
        if isinstance(v, dict) != isinstance(new[k], dict):
            raise TypeError(f"Variant dict structure at key {k} mismatched with"
                " default.")
        new[k] = update_config(new[k], v) if isinstance(v, dict) else v
    return new
