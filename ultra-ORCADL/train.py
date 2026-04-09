import os
import json
import logging
import time
from torch.utils.data import Subset

from transformers import set_seed, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_model_param_count
from torch.distributed.elastic.multiprocessing.errors import record

from dataset import DataArguments, Cmip6Dataset, ReanalyCombinedDataset
from model import ModelArguments, ORCADLConfig, ORCADLModel
from variable_config import (
    build_ocean_channel_lists,
    build_var_index,
)

from trainer import (
    Trainer, TrainingArguments,
    get_default_callbacks, setup_logger, collate_fn
)

logger = logging.getLogger(__name__)


def maybe_limit_dataset(dataset, env_name, logger, dataset_name):
    limit = os.getenv(env_name)
    if dataset is None or limit is None:
        return dataset

    limit = int(limit)
    if limit <= 0:
        return dataset

    if len(dataset) <= limit:
        logger.info("%s size=%s, %s=%s has no effect.", dataset_name, len(dataset), env_name, limit)
        return dataset

    logger.warning("Limiting %s from %s to %s samples via %s.", dataset_name, len(dataset), limit, env_name)
    return Subset(dataset, range(limit))

def setup_layer_wise_optimizer(model, base_lr=2e-5):
    """使用分层学习率替代冻结"""
    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = [
        # 编码器部分 - 极低学习率（相当于"软冻结"）
        {
            "params": [p for n, p in model.enc_ocean.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": base_lr * 0.01,  # 只有基础学习率的1%
        },
        {
            "params": [p for n, p in model.enc_atmo.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": base_lr * 0.01,
        },
        # 解码器部分 - 正常学习率
        {
            "params": [p for n, p in model.dec_ocean.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": base_lr,
        },
        # 偏置和LayerNorm - 较低学习率
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": base_lr * 0.1,
        },
    ]
    
    return torch.optim.AdamW(optimizer_grouped_parameters)

@record
def main():

    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    if data_args.data_config_path is not None:
        with open(data_args.data_config_path, 'r') as f:
            data_dict = json.load(f)
        data_args = type("DataArguments", (), data_dict)

    # Setup logging
    setup_logger(training_args, logger)

    # Log on each process the small summary:
    training_args._setup_devices

    logger.warning(
        f"Process global rank: {training_args.process_index}, local rank: {training_args.local_rank}, "
        + f"device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed: {bool(training_args.local_rank != -1)}, 16-bits: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    dataset_start_time = time.perf_counter()
    train_dataset = Cmip6Dataset(data_args, split='train')
    logger.info(
        "Train dataset initialized in %.2fs, num_samples=%s",
        time.perf_counter() - dataset_start_time,
        len(train_dataset),
    )

    eval_dataset = None
    if training_args.do_eval:
        eval_start_time = time.perf_counter()
        eval_dataset = ReanalyCombinedDataset(data_args, data_args.valid_data_dir, split='valid')
        logger.info(
            "Eval dataset initialized in %.2fs, num_samples=%s",
            time.perf_counter() - eval_start_time,
            len(eval_dataset),
        )


    var_list = train_dataset.get_input_var_list_cmip6()
    var_list = list(var_list)
    var_index = build_var_index(var_list)
    in_chans, out_chans = build_ocean_channel_lists(var_list, input_steps=data_args.input_steps)
    atmo_dims = len(train_dataset.atmo_var_list)

    if model_args.model_config_path is not None:
        logger.warning(f"Using target model config defined in {model_args.model_config_path}")
        config = ORCADLConfig.from_json_file(model_args.model_config_path)
    elif model_args.model_path is not None and os.path.isdir(model_args.model_path):
        config = ORCADLConfig.from_pretrained(model_args.model_path)
    else:
        logger.warning("Using default model config")
        config = ORCADLConfig()

    config.update({
        'var_list': var_list,
        'var_index': var_index,
        'in_chans': in_chans,
        'out_chans': out_chans,
        'max_t': data_args.max_t,
        'predict_time_steps': data_args.predict_steps,
        'atmo_dims': atmo_dims,
        'atmo_var_list': list(train_dataset.atmo_var_list),
    })
    config.update_from_args(model_args)

    train_dataset = maybe_limit_dataset(train_dataset, "MAX_TRAIN_SAMPLES", logger, "train dataset")
    eval_dataset = maybe_limit_dataset(eval_dataset, "MAX_EVAL_SAMPLES", logger, "eval dataset")

    uses_conditional_experts = any(
        bool(getattr(config, attr, False))
        for attr in ("is_moe", "is_moe_encoder", "is_moe_decoder", "is_moe_atmo")
    )
    if uses_conditional_experts and training_args.ddp_find_unused_parameters is False:
        logger.warning(
            "Detected MoE-style conditional branches in the model config; "
            "overriding ddp_find_unused_parameters=False to True for DDP correctness."
        )
        training_args.ddp_find_unused_parameters = True

    model_start_time = time.perf_counter()
    model = ORCADLModel(config)
    logger.info("Model initialized in %.2fs", time.perf_counter() - model_start_time)

    if model_args.model_path is None:
        logger.warning("Trying to train a model from scratch")
    else:
        incompatible = model.load_expanded_state_dict(model_args.model_path)
        logger.info(
            "Loaded pretrained weights with variable expansion. "
            f"missing_keys={len(incompatible.missing_keys)}, "
            f"unexpected_keys={len(incompatible.unexpected_keys)}"
        )

    logger.info(f"Model Config {model.config}")

    # 新增：冻结编码器参数
    for param in model.enc_ocean.parameters():
        param.requires_grad = False
    # 可选：也冻结大气编码器
    for param in model.enc_atmo.parameters():
        param.requires_grad = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        callbacks=get_default_callbacks(),
        data_collator=collate_fn
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["params"] = get_model_param_count(model)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    with open(os.path.join(training_args.output_dir, 'args.json'), 'w') as fp:
        json.dump({
            'data_args': data_args.to_dict(),
            'model_args': model_args.to_dict(),
            'training_args': training_args.to_dict(),
        }, fp, indent=2)


if __name__ == "__main__":
    main()
