import os
import re
import sys
import json
import torch
import logging
import collections
import transformers
import numpy as np

from collections.abc import Mapping
from typing import Dict, Any, Union, Tuple, NamedTuple, Optional
from transformers.trainer_pt_utils import _secs2timedelta


np_str_obj_array_pattern = re.compile(r'[SaUO]')

def setup_logger(args, logger):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    return logger


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    labels: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    labels: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


def metrics_format(self, metrics) -> Dict[str, Any]:
    """
    Reformat Trainer metrics values to a human-readable format

    Args:
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict

    Returns:
        metrics (`Dict[str, float]`): The reformatted metrics
    """

    if isinstance(metrics, int):
        return metrics
    elif isinstance(metrics, float):
        return round(metrics, 4)
    
    metrics_copy = metrics.copy()
    if isinstance(metrics_copy, dict):
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{ v >> 20 }MB"
            elif "_runtime" in k:
                metrics_copy[k] = _secs2timedelta(v)
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            else:
                metrics_copy[k] = self.metrics_format(metrics_copy[k])

    elif isinstance(metrics_copy, list):
        for i in range(len(metrics_copy)):
            metrics_copy[i] = self.metrics_format(metrics_copy[i])

    return metrics_copy


def log_metrics(self, split, metrics):
    """
    Log metrics in a specially formatted way

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predictmetrics: metrics dict
    """
    if not self.is_world_process_zero():
        return

    print(f"***** {split} metrics *****")
    metrics_formatted = self.metrics_format(metrics)
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    # v_width = max(len(str(x)) for x in metrics_formatted.values())
    for key in sorted(metrics_formatted.keys()):
        print(f"  {key: <{k_width}} = {metrics_formatted[key]}") #:<{v_width}


def save_metrics(self, split, metrics, combined=True):
    """
    Save metrics into a json file for that split, e.g. `train_results.json`.

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`, `all`
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict
        combined (`bool`, *optional*, defaults to `True`):
            Creates combined metrics by updating `all_results.json` with metrics of this call

    To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
    unformatted numbers are saved in the current method.

    """
    if not self.is_world_process_zero():
        return

    path = os.path.join(self.args.output_dir, f"{split}_results.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    if combined:
        path = os.path.join(self.args.output_dir, f"{split}_results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics = recursive_update_dict(all_metrics, metrics)
        with open(path, "w") as f:
            json.dump(all_metrics, f, indent=4, sort_keys=True)


def compute_all_metrics(all_metrics):
    elem = all_metrics[0]
    if isinstance(elem, (float, int)):
        return np.mean(all_metrics)
    elif isinstance(elem, (list, tuple)):
        return type(elem)(np.stack(all_metrics).mean(0).tolist())
    elif isinstance(elem, Mapping):
        return {k: compute_all_metrics([m[k] for m in all_metrics]) for k in elem.keys()}


def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(f"collate type error, type {elem.dtype}")
            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, (int, bool)):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate_fn([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate_fn(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate_fn(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate_fn(samples) for samples in transposed]

    raise TypeError(f"collate type error, type {elem.dtype}")


def default_collate_fn(batch):
    elem = batch[0]
    
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    
    elif isinstance(elem, Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}

    assert 0

# https://github.com/Maples7/dict-recursive-update/tree/master
def recursive_update_dict(default, custom):
    '''Return a dict merged from default and custom

    >>> recursive_update('a', 'b')
    Traceback (most recent call last):
        ...
    TypeError: Params of recursive_update should be dicts

    >>> recursive_update({'a': [1]}, {'a': [2], 'c': {'d': {'c': 3}}})
    {'a': [2], 'c': {'d': {'c': 3}}}

    >>> recursive_update({'a': {'c': 1, 'd': {}}, 'b': 4}, {'b': 5})
    {'a': {'c': 1, 'd': {}}, 'b': 5}

    >>> recursive_update({'a': {'c': 1, 'd': {}}, 'b': 4}, {'a': 2})
    {'a': 2, 'b': 4}
    '''
    if not isinstance(default, dict) or not isinstance(custom, dict):
        raise TypeError('Params of recursive_update should be dicts')

    for key in custom:
        if isinstance(custom[key], dict) and isinstance(default.get(key), dict):
            default[key] = recursive_update_dict(default[key], custom[key])
        else:
            default[key] = custom[key]

    return default