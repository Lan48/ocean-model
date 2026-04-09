import os
import json
import logging
import torch
import torch.nn as nn

from transformers import set_seed, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_model_param_count

from dataset import DataArguments, Cmip6Dataset, ReanalyCombinedDataset
from model import ModelArguments, ORCADLConfig, ORCADLModel

from trainer import (
    Trainer, TrainingArguments,
    get_default_callbacks, setup_logger, collate_fn
)

logger = logging.getLogger(__name__)

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

    train_dataset = Cmip6Dataset(data_args, split='train')

    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = ReanalyCombinedDataset(data_args, data_args.valid_data_dir, split='valid')


    var_list = train_dataset.get_input_var_list_cmip6()
    var_index = [train_dataset.get_var_index(v) for v in var_list]

    if model_args.model_path is None:
        logger.warning("Trying to train a model from scratch")
        if model_args.model_config_path is not None:
            logger.warning(f"Using model config defined in {model_args.model_config_path}")
            config = ORCADLConfig.from_json_file(model_args.model_config_path)
        else:
            logger.warning("Using default model config")
            config = ORCADLConfig()

        config.update({
            'var_list': var_list,
            'var_index': var_index,
            'max_t': data_args.max_t,
            'predict_time_steps': data_args.predict_steps,
        })
        config.update_from_args(model_args)

        model = ORCADLModel(config)
    else:
        # 支持 model_path 指向目录（包含 config.json + pytorch_model.bin）或指向单个 .bin 权重文件
        if os.path.isfile(model_args.model_path):
            # 如果给的是单个权重文件，优先使用显式提供的 model_config_path 加载配置
            if model_args.model_config_path is not None:
                config = ORCADLConfig.from_json_file(model_args.model_config_path)
            else:
                config = ORCADLConfig()

            config.update_from_args(model_args)
            model = ORCADLModel(config)

            # 手动加载 state_dict，只覆盖键名存在且形状匹配的参数
            sd = torch.load(model_args.model_path, map_location='cpu')
            if 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            new_sd = {}
            for k, v in sd.items():
                key = k.replace('module.', '', 1) if k.startswith('module.') else k
                new_sd[key] = v
            model_state = model.state_dict()
            matched = {k: v for k, v in new_sd.items() if k in model_state and v.size() == model_state[k].size()}
            model_state.update(matched)
            model.load_state_dict(model_state)
        else:
            # model_path 是目录，尝试手动在目录下查找 config.json / model_config.json 和 pytorch_model.bin
            config_file = None
            for cand in ('config.json', 'model_config.json'):
                cand_path = os.path.join(model_args.model_path, cand)
                if os.path.exists(cand_path):
                    config_file = cand_path
                    break

            if config_file is not None:
                config = ORCADLConfig.from_json_file(config_file)
            else:
                # 回退到 from_pretrained 读取（会报错如果没有 config）
                config = ORCADLConfig.from_pretrained(model_args.model_path)

            config.update_from_args(model_args)
            model = ORCADLModel(config)

            # 在目录下寻找权重文件
            bin_file = os.path.join(model_args.model_path, 'pytorch_model.bin')
            if not os.path.exists(bin_file):
                # 尝试常见备选名
                for fname in os.listdir(model_args.model_path):
                    if fname.endswith('.bin'):
                        bin_file = os.path.join(model_args.model_path, fname)
                        break

            if os.path.exists(bin_file):
                sd = torch.load(bin_file, map_location='cpu')
                if 'model_state_dict' in sd:
                    sd = sd['model_state_dict']
                new_sd = {}
                for k, v in sd.items():
                    key = k.replace('module.', '', 1) if k.startswith('module.') else k
                    new_sd[key] = v
                model_state = model.state_dict()
                matched = {k: v for k, v in new_sd.items() if k in model_state and v.size() == model_state[k].size()}
                model_state.update(matched)
                model.load_state_dict(model_state)
            else:
                # 如果找不到 bin 文件，尝试使用 transformers 的 from_pretrained 作为最后手段
                model = ORCADLModel.from_pretrained(model_args.model_path, config=config,
                                                   ignore_mismatched_sizes=model_args.ignore_mismatched_sizes)

        # 加载完成后，若需要保证第二分支初始输出为0，重新 zero-init 第二分支模块
        def _zero_init_module(mod: nn.Module):
            for p in mod.parameters():
                if p is not None:
                    p.data.zero_()
            for _name, buf in mod.named_buffers():
                if isinstance(buf, torch.Tensor):
                    try:
                        buf.data.zero_()
                    except Exception:
                        pass

        _zero_init_module(model.enc_ocean2)
        _zero_init_module(model.fusion2)
        _zero_init_module(model.dec_ocean2)
        _zero_init_module(model.enc_atmo2)

        model.config.update({
            'predict_time_steps': data_args.predict_steps,
        })

    logger.info(f"Model Config {model.config}")

    # 冻结指定子模块参数（不训练）
    for param in model.enc_ocean.parameters():
        param.requires_grad = False
    for param in model.fusion.parameters():
        param.requires_grad = False
    for param in model.dec_ocean.parameters():
        param.requires_grad = False
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