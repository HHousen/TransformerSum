import os
import logging
import torch
from pytorch_lightning import Trainer
from model import ExtractiveSummarizer
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def main(args):
    model = ExtractiveSummarizer(hparams=args)
    if args.load_weights:
        checkpoint = torch.load(
            args.load_weights, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["state_dict"])

    if args.use_logger == "wandb":
        wandb_logger = WandbLogger(project="transformerextsum", log_model=True)
        args.logger = wandb_logger
        models_path = os.path.join(wandb_logger.experiment.dir, "models/")
    else:
        models_path = "models/"

    if args.use_custom_checkpoint_callback:
        args.checkpoint_callback = ModelCheckpoint(
            filepath=models_path, save_top_k=-1, period=1, verbose=True,
        )

    trainer = Trainer.from_argparse_args(args)
    if args.do_train:
        trainer.fit(model)
    if args.do_test:
        trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # parametrize the network: general options
    parser.add_argument(
        "--default_save_path", type=str, help="Default path for logs and weights",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,  # 2e-3 and 5e-5 and 3e-6
        type=float,
        help="The initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--min_epochs",
        default=3,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Limits training to a max number number of epochs",
    )
    parser.add_argument(
        "--min_steps",
        default=0,
        type=int,
        help="Limits training to a minimum number number of steps",
    )
    parser.add_argument(
        "--max_steps",
        default=1_000_000,
        type=int,
        help="Limits training to a max number number of steps",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="Accumulates grads every k batches or as set up in the dict.",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        default=1,
        type=int,
        help="Check val every n train epochs.",
    )
    parser.add_argument(
        "--gpus",
        default=-1,
        type=int,
        help="Number of GPUs to train on or Which GPUs to train on. (default: -1 (all gpus))",
    )
    parser.add_argument(
        "--gradient_clip_val", default=1.0, type=float, help="Gradient clipping value"
    )
    parser.add_argument(
        "--overfit_pct",
        default=0.0,
        type=float,
        help="Uses this much data of all datasets (training, validation, test). Useful for quickly debugging or trying to overfit on purpose.",
    )
    parser.add_argument(
        "--amp_level",
        type=str,
        default="O1",
        help="The optimization level to use (O1, O2, etc…) for 16-bit GPU precision (using NVIDIA apex under the hood).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.",
    )
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="To profile individual steps during training and assist in identifying bottlenecks.",
    )
    parser.add_argument(
        "--progress_bar_refresh_rate",
        default=50,
        type=int,
        help="How often to refresh progress bar (in steps). In notebooks, faster refresh rates (lower number) is known to crash them because of their screen refresh rates, so raise it to 50 or more.",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        default=5,
        type=int,
        help="Sanity check runs n batches of val before starting the training routine. This catches any bugs in your validation without having to wait for the first validation check.",
    )
    parser.add_argument(
        "--use_logger",
        default="wandb",
        type=str,
        choices=["tensorboard", "wandb"],
        help="Which program to use for logging. If `--use_custom_checkpoint_callback` is specified and `wandb` is chosen then model weights will automatically be uploaded to wandb.ai.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Run the training procedure."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Run the testing procedure."
    )
    parser.add_argument(
        "--load_weights",
        default=False,
        type=str,
        help="Loads the model weights from a given checkpoint",
    )
    parser.add_argument(
        "--use_custom_checkpoint_callback",
        action="store_true",
        help="""Use the custom checkpointing callback specified in main() by 
        `args.checkpoint_callback`. By default this custom callback saves the model every 
        epoch and never deletes and saved weights files. Set this option and `--use_logger` 
        to `wandb` to automatically upload model weights to wandb.ai. DO NOT set this and 
        `--user_logger` to "tensorboard" because a custom TensorBoardLogger is not created. 
        Thus, when the trainer attempts to save the model, the program will crash since
        `--default_save_path` is set and a custom checkpoint callback is passed.
        See: https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#default-root-dir
        ("Default path for logs and weights when **no logger or ModelCheckpoint callback** passed.")""",
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )

    parser = ExtractiveSummarizer.add_model_specific_args(parser)

    args = parser.parse_args()

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(args.logLevel),
    )

    # Train
    main(args)
