import numpy as np
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerControl, TrainerCallback


def get_default_callbacks():
    return [FlowCallback]


class FlowCallback(TrainerCallback):

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        # Ensure that the model is saved when evaluation results in better scores
        metrics = kwargs.pop('metrics', None)
        if args.load_best_model_at_end and metrics is not None:
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            metric_value = metrics[metric_to_check]
            operator = np.greater if args.greater_is_better else np.less
            if (
                state.best_metric is None
                or state.best_model_checkpoint is None
                or operator(metric_value, state.best_metric)
            ):
                control.should_save = True

        return control

    # Make sure to do evaluation at the end of each epoch
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        if args.do_eval and args.load_best_model_at_end and np.fabs(state.num_train_epochs - state.epoch) <= 1e-3:
            control.should_evaluate = True
        return control