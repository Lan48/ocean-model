import os
import json
import logging

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
        config = ORCADLConfig.from_pretrained(model_args.model_path)
        config.update_from_args(model_args)
        model = ORCADLModel.from_pretrained(model_args.model_path, config=config,
                                          ignore_mismatched_sizes=model_args.ignore_mismatched_sizes)
        model.config.update({
            'predict_time_steps': data_args.predict_steps,
        })

    logger.info(f"Model Config {model.config}")


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