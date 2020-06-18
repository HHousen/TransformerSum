import os
import json
import gzip
import logging
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


def load_json(json_file):
    # `file_extension` is second and path (without extension) is first
    # `file_extension` only contains last extension so ".json.gz" will output ".gz"
    file_path, file_extension = os.path.splitext(json_file)
    if file_extension == ".json":
        with open(json_file, "r") as json_file_object:
            documents = json.load(json_file_object)
    elif file_extension == ".gz":
        file_path = os.path.splitext(file_path)[0]  # remove ".gz"
        # https://stackoverflow.com/a/39451012
        with gzip.open(json_file, "r") as json_gzip:
            json_bytes = json_gzip.read()
        json_str = json_bytes.decode("utf-8")
        documents = json.loads(json_str)  # "loads": the "s" means string
    else:
        logger.error(
            "File extension "
            + str(file_extension)
            + " not recognized. Please use either '.json' or '.gz'."
        )
    return documents, file_path


class StepCheckpointCallback(pl.callbacks.base.Callback):
    def __init__(self, step_interval=1000, save_name="model", save_path=".", num_saves_to_keep=5):
        super(StepCheckpointCallback, self).__init__()
        self.step_interval = step_interval
        self.save_name = save_name
        self.save_path = save_path
        self.num_saves_to_keep = num_saves_to_keep

    def on_batch_end(self, trainer, pl_module):
        # check if `step_interval` has passed and that the `global_step` is not 0
        if trainer.global_step % self.step_interval == 0 and not trainer.global_step == 0:
            final_save_location = os.path.join(
                self.save_path, (self.save_name + "." + str(trainer.global_step) + ".ckpt")
            )
            trainer.save_checkpoint(final_save_location)
            # remove previous saves
            offset = self.step_interval * self.num_saves_to_keep
            path_to_remove = self.save_name + "." + str(trainer.global_step-offset) + ".ckpt"
            if os.path.isfile(path_to_remove):
                os.remove(path_to_remove)

def lr_lambda_func(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    )