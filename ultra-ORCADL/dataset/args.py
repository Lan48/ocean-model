import json
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field, fields

@dataclass
class DataArguments:
    data_dir: str = field(
        default=None, metadata={"help": "Data dir"}
    )
    valid_data_dir: List[str] = field(
        default_factory=list,
    )
    data_config_path: str = field(
        default=None, metadata={"help": "Data config path"}
    )
    input_var_list : List[str] = field(
        default_factory=list, metadata={"help": "The variables used, if not set, will find all the available variables in data dir"}
    )
    var_list_not_used: List[str] = field(
        default_factory=list, metadata={"help": "The variables not used in data dir"}
    )
    cached: bool = field(
        default=False, metadata={"help": "Whether the data is cahced"}
    )
    do_cache: bool = field(
        default=False, metadata={"help": "Whether to cache data "}
    )
    cache_sample_dir: Optional[str] = field(
        default="./cache", metadata={"help": "Data cache dir"}
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Whether to overwrite cache dir"}
    )
    check_data: bool = field(
        default=False, metadata={"help": "Whether to check the data after loading"}
    )
    train_val_test_split_by: Optional[str] = field(
        default='model', metadata={"help": "How to split dataset.", "choices":['model', 'time']}
    )
    train_val_split_year: Optional[int] = field(
        default=None, metadata={"help": "The split year of the training data and the validation data"}
    )
    val_test_split_year: Optional[int] = field(
        default=None, metadata={"help": "The split year of the validation data and the testing data"}
    )
    train_model_list: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "The models used for training, if not specified,"
                                         "will use all models excecpt that included in val_model_list and test_model_list"}
    )
    val_model_list: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "The models used for validation"}
    )
    test_model_list: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "The models used for testing"}
    )
    model_list_not_used: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "The models not used"}
    )
    input_steps: int = field(
        default=1, metadata={"help": "How many time steps used to predict next steps"}
    )
    predict_steps: int = field(
        default=1, metadata={"help": "How many time steps to predict"}
    )
    atmo_var_list: List[str] = field(
        default_factory=list, metadata={"help": "The atmo variables used"}
    )
    max_t: int = field(
        default=None
    )
    lead_t_weight: bool = field(
        default=False
    )
    max_t_weight: float = field(
        default=0.7
    )
    min_t_weight: float = field(
        default=0.3
    )
    lead_t: int = field(
        default=None
    )
    valid_lead_t: int = field(
        default=1
    )
    start_year: int = field(
        default=None
    )
    end_year: int = field(
        default=None
    )
    use_atmo: bool = field(
        default=True
    )
    valid_all_t: bool = field(
        default=False
    )
    
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

