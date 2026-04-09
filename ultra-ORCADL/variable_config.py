from __future__ import annotations

from typing import Iterable, List, Sequence


VARIABLE_SPECS = {
    "so": {"channels": 16, "level": "multi", "stat_name": "salt"},
    "thetao": {"channels": 16, "level": "multi", "stat_name": "pottmp"},
    "tos": {"channels": 1, "level": "single", "stat_name": "sst"},
    "uo": {"channels": 16, "level": "multi", "stat_name": "ucur"},
    "vo": {"channels": 16, "level": "multi", "stat_name": "vcur"},
    "zos": {"channels": 1, "level": "single", "stat_name": "sshg"},
    "hfds": {"channels": 1, "level": "single", "stat_name": "hfds"},
    "mlotst": {"channels": 1, "level": "single", "stat_name": "mlotst"},
    "rsntds": {"channels": 1, "level": "single", "stat_name": "rsntds"},
    "sob": {"channels": 1, "level": "single", "stat_name": "sob"},
    "sos": {"channels": 1, "level": "single", "stat_name": "sos"},
    "tob": {"channels": 1, "level": "single", "stat_name": "tob"},
    "wfo": {"channels": 1, "level": "single", "stat_name": "wfo"},
    "wo": {"channels": 16, "level": "multi", "stat_name": "wo"},
    "tauu": {"channels": 1, "level": "single", "stat_name": "uflx"},
    "tauv": {"channels": 1, "level": "single", "stat_name": "vflx"},
}

DEFAULT_MODEL_OCEAN_VARS = [
    "so",
    "thetao",
    "tos",
    "uo",
    "vo",
    "zos",
    "hfds",
    "mlotst",
    "rsntds",
    "sob",
    "sos",
    "tob",
    "wfo",
    "wo",
]

DEFAULT_MODEL_ATMO_VARS = ["tauu", "tauv"]

MODEL_VAR_ORDER = DEFAULT_MODEL_OCEAN_VARS + DEFAULT_MODEL_ATMO_VARS
MODEL_TO_STAT_MAPPING = {
    var_name: spec["stat_name"] for var_name, spec in VARIABLE_SPECS.items()
}
STAT_TO_MODEL_MAPPING = {
    stat_name: var_name for var_name, stat_name in MODEL_TO_STAT_MAPPING.items()
}

MODEL_SINGLE_LEVEL_VARS = {
    var_name for var_name, spec in VARIABLE_SPECS.items() if spec["level"] == "single"
}
MODEL_MULTI_LEVEL_VARS = {
    var_name for var_name, spec in VARIABLE_SPECS.items() if spec["level"] == "multi"
}
STAT_SINGLE_LEVEL_VARS = {
    MODEL_TO_STAT_MAPPING[var_name] for var_name in MODEL_SINGLE_LEVEL_VARS
}
STAT_MULTI_LEVEL_VARS = {
    MODEL_TO_STAT_MAPPING[var_name] for var_name in MODEL_MULTI_LEVEL_VARS
}
ALL_SINGLE_LEVEL_VARS = MODEL_SINGLE_LEVEL_VARS | STAT_SINGLE_LEVEL_VARS
ALL_MULTI_LEVEL_VARS = MODEL_MULTI_LEVEL_VARS | STAT_MULTI_LEVEL_VARS


def normalize_to_model_var(var_name: str) -> str:
    return STAT_TO_MODEL_MAPPING.get(var_name, var_name)


def to_stat_var(var_name: str) -> str:
    return MODEL_TO_STAT_MAPPING.get(normalize_to_model_var(var_name), var_name)


def is_supported_var(var_name: str) -> bool:
    return normalize_to_model_var(var_name) in VARIABLE_SPECS


def ensure_supported_vars(var_names: Iterable[str]) -> List[str]:
    normalized = [normalize_to_model_var(var_name) for var_name in var_names]
    unknown = [var_name for var_name in normalized if var_name not in VARIABLE_SPECS]
    if unknown:
        raise ValueError(f"Unsupported variables: {unknown}")
    return normalized


def _deduplicate(var_names: Sequence[str]) -> List[str]:
    seen = set()
    deduped = []
    for var_name in var_names:
        if var_name not in seen:
            deduped.append(var_name)
            seen.add(var_name)
    return deduped


def sort_vars_by_registry(var_names: Sequence[str], registry_order: Sequence[str]) -> List[str]:
    normalized = _deduplicate(ensure_supported_vars(var_names))
    order_index = {var_name: idx for idx, var_name in enumerate(registry_order)}
    return sorted(normalized, key=lambda var_name: order_index.get(var_name, len(order_index)))


def sort_ocean_vars(var_names: Sequence[str]) -> List[str]:
    return [var_name for var_name in sort_vars_by_registry(var_names, DEFAULT_MODEL_OCEAN_VARS)]


def sort_atmo_vars(var_names: Sequence[str]) -> List[str]:
    return [var_name for var_name in sort_vars_by_registry(var_names, DEFAULT_MODEL_ATMO_VARS)]


def get_var_channels(var_name: str) -> int:
    return int(VARIABLE_SPECS[normalize_to_model_var(var_name)]["channels"])


def is_single_level_var(var_name: str) -> bool:
    return normalize_to_model_var(var_name) in MODEL_SINGLE_LEVEL_VARS


def is_multi_level_var(var_name: str) -> bool:
    return normalize_to_model_var(var_name) in MODEL_MULTI_LEVEL_VARS


def build_ocean_channel_lists(var_names: Sequence[str], input_steps: int = 1) -> tuple[list[int], list[int]]:
    ocean_vars = sort_ocean_vars(var_names)
    out_chans = [get_var_channels(var_name) for var_name in ocean_vars]
    in_chans = [channels * input_steps for channels in out_chans]
    return in_chans, out_chans


def build_var_index(var_names: Sequence[str]) -> List[int]:
    ocean_vars = sort_ocean_vars(var_names)
    single_vars = [var_name for var_name in ocean_vars if is_single_level_var(var_name)]
    multi_vars = [var_name for var_name in ocean_vars if is_multi_level_var(var_name)]

    var_index = []
    for var_name in ocean_vars:
        if is_single_level_var(var_name):
            var_index.append(single_vars.index(var_name))
        else:
            var_index.append(multi_vars.index(var_name))
    return var_index


def infer_input_steps(in_chans: Sequence[int], out_chans: Sequence[int]) -> int:
    if len(in_chans) != len(out_chans):
        raise ValueError("in_chans and out_chans must have the same length")
    ratios = {
        int(in_chan // out_chan)
        for in_chan, out_chan in zip(in_chans, out_chans)
        if out_chan > 0 and in_chan % out_chan == 0
    }
    if not ratios:
        raise ValueError("Unable to infer input_steps from empty channel lists")
    if len(ratios) != 1:
        raise ValueError(f"Inconsistent input/output channel ratios: {sorted(ratios)}")
    return ratios.pop()

