import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List
from itertools import accumulate
from dataclasses import dataclass, field
from transformers import set_seed, HfArgumentParser
from torch.utils.data import DataLoader

from trainer import (
    TrainingArguments,
    setup_logger,
    collate_fn,
)

from dataset import DataArguments, GodasDataset
from model import ORCADLModel, ORCADLConfig, ModelArguments


logger = logging.getLogger(__name__)

import json
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field, fields


@dataclass
class TestingArguments(TrainingArguments):

    num_test_samples: int = field(
        default=None, metadata={"help": "How many samples used for testing."}
    )
    test_data_indices: List[int] = field(
        default_factory=list, metadata={"help": "Samples indices in test dataset. If set, will ignore num test samples."}
    )
    save_preds: bool = field(
        default=False, metadata={"help": "If True, will save model predictions and true labels."}
    )
    test_start_year: int = field(
        default=1983
    )
    test_end_year: int = field(
        default=2020
    )
    pred_start_year: int = field(
        default=1980
    )
    ckpt_list: List[str] = field(
        default_factory=list
    )
    config_path_list: List[str] = field(
        default_factory=list
    )
    save_vars: List[str] = field(
        default_factory=list
    )

    def __post_init__(self):
        super().__post_init__()        
        if len(self.test_data_indices) > 2:
            raise ValueError("Indices should be two values at most.([start, end) or [: end))")


def main():

    parser = HfArgumentParser((TestingArguments, DataArguments, ModelArguments))
    testing_args, data_args, model_args = parser.parse_args_into_dataclasses()

    # Setup logging
    setup_logger(testing_args, logger)

    # Log on each process the small summary:
    # logger.warning(
    #     f"Process global rank: {testing_args.global_rank}, local rank: {testing_args.local_rank}, "
    #     + f"device: {testing_args.device}, n_gpu: {testing_args.n_gpu}, "
    #     + f"distributed: {bool(testing_args.local_rank != -1)}, 16-bits: {testing_args.fp16}"
    # )
    logger.info(f"Testing parameters {testing_args}")

    # Set seed before initializing model.
    set_seed(testing_args.seed)


    test_dataset = GodasDataset(data_args)

    var_list = test_dataset.get_input_var_list_cmip6()
    var_index = [test_dataset.get_var_index(v) for v in var_list]


    model_list = []

    if len(testing_args.ckpt_list) != len(testing_args.config_path_list):
        if len(testing_args.config_path_list) == 1:
            testing_args.config_path_list = [testing_args.config_path_list[0]] * len(testing_args.ckpt_list)
            logger.warning(
                "Only one config path is provided, using it for all checkpoints."
            )
        else:
            raise ValueError("The config path list length should be the same as the checkpoint list length or 1.")

    for ckpt_path, cfg_path in zip(testing_args.ckpt_list, testing_args.config_path_list):

        config = ORCADLConfig.from_json_file(cfg_path)
        model = ORCADLModel(config)
        model.load_state_dict(torch.load(ckpt_path))
    
        if model.config.var_list != var_list or model.config.var_index != var_index:
            raise ValueError("var_list/var_index in args is not the same as in pretrained model config")

        model.config.update({
            'predict_time_steps': data_args.predict_steps,
        })
        model_list.append(model)


    indices_ = None
    if testing_args.num_test_samples is not None or len(testing_args.test_data_indices) > 0:
        indices = testing_args.test_data_indices
        if len(indices) > 0:
            if len(indices) == 1:
                indices_ = list(range(indices[0]))
            else:
                if indices[1] < 0:
                    indices[1] = len(test_dataset) + indices[1] + 1
                indices_ = list(range(indices[0], indices[1]))
        else:
            indices_ = list(range(testing_args.num_test_samples))


    if indices_ is not None:
        test_dataset = test_dataset.get_subset(indices_)

    init_time_list = test_dataset.times[:len(test_dataset)]

    dataloader = DataLoader(dataset=test_dataset,
                            batch_size=testing_args.per_device_eval_batch_size,
                            shuffle=False,
                            num_workers=testing_args.dataloader_num_workers,
                            collate_fn=collate_fn,
                            drop_last=False)
    for model in model_list:
        model.cuda()
        model.eval()

    start_months = []
    preds = []
    labels = []

    # print(pred_month.shape, pred_month[:2])
    model_config = model.config
    results ={}

    outdir = testing_args.output_dir
    preds_save_dir = os.path.join(testing_args.output_dir, 'preds')
    os.makedirs(outdir, exist_ok=True)
    if testing_args.save_preds:
        os.makedirs(preds_save_dir, exist_ok=True)

    num_model = len(model_list)

    for batch in tqdm(dataloader):
        start_months.append(batch.pop('start_month').numpy())
        batch.pop('labels')
        # labels.append(batch.pop('labels').numpy())
        for k in batch:
            batch[k] = batch[k].cuda()
        # batch['ocean_vars'] = batch['ocean_vars'].cuda()
        with torch.no_grad():
            logits_sum = 0.0
            for model in model_list:
                logits = model(**batch).preds
                logits_sum += logits
            logits = logits_sum / num_model  # ensemble mean prediction
                
        preds.append(logits.detach().cpu().numpy())
    
    start_months = np.concatenate(start_months, axis=0)

    split_indices = np.array(model_config.out_chans)

    split_indices = list(accumulate(split_indices))[:-1]
    split_axis = 1 if data_args.predict_steps == 1 else 2

    preds = [np.split(p, split_indices, axis=split_axis) for p in preds]

    all_preds = []
    for i in range(len(preds[0])):
        all_preds.append(np.concatenate([p[i] for p in preds], axis=0))

    for v, pred in zip(var_list, all_preds):

        pred = pred.squeeze()
        # label = label.squeeze()
        mask = np._NoValue

        if testing_args.save_preds and v in testing_args.save_vars:
            if data_args.predict_steps == 1:
                torch.save(pred, os.path.join(preds_save_dir, f'{v}.pt'), pickle_protocol=4)
            else:
                for i in range(pred.shape[1]):
                    torch.save(pred[:, i], os.path.join(preds_save_dir, f'{v}_step{i+1}.pt'), pickle_protocol=4)
    
    json.dump(init_time_list, open(os.path.join(outdir, 'init_times.json'), 'w'), indent=4)

    print('\n' + '*'*10 + ' Done ' + '*'*10)

if __name__ == "__main__":
    main()