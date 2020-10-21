import os
import logging
import torch
import numpy as np
import random
import datasets as nlp
from pytorch_lightning import Trainer
from extractive import ExtractiveSummarizer
from abstractive import AbstractiveSummarizer
from helpers import StepCheckpointCallback
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.warning(
        "Deterministic mode can have a performance impact, depending on your model. This means that due to the deterministic nature of the model, the processing speed (i.e. processed batch items per second) can be lower than when the model is non-deterministic."
    )


def main(args):
    if args.seed:
        set_seed(args.seed)

    if args.mode == "abstractive":
        summarizer = AbstractiveSummarizer
    else:
        summarizer = ExtractiveSummarizer

    if args.load_weights:
        model = summarizer(hparams=args)
        checkpoint = torch.load(
            args.load_weights, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["state_dict"])
    elif args.load_from_checkpoint:
        model = summarizer.load_from_checkpoint(args.load_from_checkpoint)
        # The model is loaded with self.hparams.data_path set to the directory where the data
        # was located during training. When loading the model, it may be desired to change
        # the data path, which the below line accomplishes.
        if args.data_path:
            model.hparams.data_path = args.data_path
        # Same as above but for `test_use_pyrouge`
        if args.test_use_pyrouge:
            model.hparams.test_use_pyrouge = args.test_use_pyrouge
    else:
        model = summarizer(hparams=args)

    # Create learning rate logger
    lr_logger = LearningRateMonitor()
    args.callbacks = [lr_logger]

    if args.use_logger == "wandb":
        wandb_logger = WandbLogger(
            project=args.wandb_project,
        )
        args.logger = wandb_logger
        if not args.no_wandb_logger_log_model:
            import wandb

            args.weights_save_path = wandb_logger.experiment.dir
            checkpoint_glob = os.path.join(wandb_logger.experiment.dir, "*.ckpt")
            wandb.save(checkpoint_glob)

    if args.use_custom_checkpoint_callback:
        args.checkpoint_callback = ModelCheckpoint(
            filepath=args.weights_save_path,
            save_top_k=-1,
            period=1,
            verbose=True,
        )
    if args.custom_checkpoint_every_n:
        custom_checkpoint_callback = StepCheckpointCallback(
            step_interval=args.custom_checkpoint_every_n,
            save_path=args.weights_save_path,
        )
        args.callbacks.append(custom_checkpoint_callback)

    trainer = Trainer.from_argparse_args(args)

    if args.lr_find:
        lr_finder = trainer.lr_find(model)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        logger.info("Recommended Learning Rate: %s", new_lr)

    # remove `args.callbacks` if it exists so it does not get saved with the model (would result in crash)
    if args.custom_checkpoint_every_n:
        del args.callbacks

    if args.do_train:
        trainer.fit(model)
    if args.do_test:
        trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # parametrize the network: general options
    parser.add_argument(
        "--mode",
        type=str,
        default="extractive",
        choices=["extractive", "abstractive"],
        help="Extractive or abstractive summarization training. Default is 'extractive'.",
    )
    parser.add_argument(
        "--default_root_dir",
        type=str,
        help="Default path for logs and weights. To use this option with the `wandb` logger specify the `--no_wandb_logger_log_model` option.",
    )
    parser.add_argument(
        "--weights_save_path",
        type=str,
        help="""Where to save weights if specified. Will override `--default_root_dir` for 
        checkpoints only. Use this if for whatever reason you need the checkpoints stored in 
        a different place than the logs written in `--default_root_dir`. This option will 
        override the save locations when using a custom checkpoint callback, such as those
        created when using `--use_custom_checkpoint_callback or `--custom_checkpoint_every_n`.
        If you are using the `wandb` logger, then you must also set `--no_wandb_logger_log_model`
        when using this option. Model weights are saved with the wandb logs to be uploaded to 
        wandb.ai by default. Setting this option without setting `--no_wandb_logger_log_model` 
        effectively creates two save paths, which will crash the script.""",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,  # 2e-3 and 5e-5 and 3e-6
        type=float,
        help="The initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
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
        default=None,
        type=int,
        help="Limits training to a minimum number number of steps",
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Limits training to a max number number of steps",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="""Accumulates grads every k batches. A single step is one gradient accumulation cycle,
        so setting this value to 2 will cause 2 batches to be processed for each step.""",
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
        "--train_percent_check",
        default=1.0,
        type=float,
        help="How much of training dataset to check. Useful when debugging or testing something that happens at the end of an epoch.",
    )
    parser.add_argument(
        "--val_percent_check",
        default=1.0,
        type=float,
        help="How much of validation dataset to check. Useful when debugging or testing something that happens at the end of an epoch.",
    )
    parser.add_argument(
        "--test_percent_check",
        default=1.0,
        type=float,
        help="How much of test dataset to check.",
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
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible results. Can negatively impact performace in some cases.",
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
        help="Which program to use for logging. If `wandb` is chosen then model weights will automatically be uploaded to wandb.ai.",
    )
    parser.add_argument(
        "--wandb_project",
        default="transformerextsum-private",
        type=str,
        help="The wandb project to save training runs to if `--use_logger` is set to `wandb`.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (save memory at the expense of a slower backward pass) for the word embedding model. More info: https://github.com/huggingface/transformers/pull/4659#issue-424841871",
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
        "--load_from_checkpoint",
        default=False,
        type=str,
        help="Loads the model weights and hyperparameters from a given checkpoint.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str,
        help="To resume training from a specific checkpoint pass in the path here. Automatically restores model, epoch, step, LR schedulers, apex, etc...",
    )
    parser.add_argument(
        "--use_custom_checkpoint_callback",
        action="store_true",
        help="""Use the custom checkpointing callback specified in `main()` by 
        `args.checkpoint_callback`. By default this custom callback saves the model every 
        epoch and never deletes the saved weights files. You can change the save path by 
        setting the `--weights_save_path` option.""",
    )
    parser.add_argument(
        "--custom_checkpoint_every_n",
        type=int,
        default=None,
        help="""The number of steps between additional checkpoints. By default checkpoints are saved 
        every epoch. Setting this value will save them every epoch and every N steps. This does not 
        use the same callback as `--use_custom_checkpoint_callback` but instead uses a different class 
        called `StepCheckpointCallback`. You can change the save path by setting the 
        `--weights_save_path` option.""",
    )
    parser.add_argument(
        "--no_wandb_logger_log_model",
        action="store_true",
        help="Only applies when using the `wandb` logger. Set this argument to NOT save checkpoints in wandb directory to upload to W&B servers.",
    )
    parser.add_argument(
        "--auto_scale_batch_size",
        default=None,
        type=str,
        help="""Auto scaling of batch size may be enabled to find the largest batch size that fits 
        into memory. Larger batch size often yields better estimates of gradients, but may also 
        result in longer training time. Currently, this feature supports two modes 'power' scaling 
        and 'binsearch' scaling. In 'power' scaling, starting from a batch size of 1 keeps doubling 
        the batch size until an out-of-memory (OOM) error is encountered. Setting the argument to 
        'binsearch' continues to finetune the batch size by performing a binary search. 'binsearch'
        is the recommended option.""",
    )
    parser.add_argument(
        "--lr_find",
        action="store_true",
        help="Runs a learning rate finder algorithm (see https://arxiv.org/abs/1506.01186) before any training, to find optimal initial learning rate.",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        help="""Which optimizer to use: `adamw` (default), `ranger`, `qhadam`, `radam`, or `adabound`.""",
    )
    parser.add_argument(
        "--ranger-k",
        default=6,
        type=int,
        help="""Ranger (LookAhead) optimizer k value (default: 6). LookAhead keeps a single
        extra copy of the weights, then lets the internalized ‘faster’ optimizer (for Ranger,
        that’s RAdam) explore for 5 or 6 batches. The batch interval is specified via the k parameter.""",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. Only active if `--use_scheduler` is set to linear.",
    )
    parser.add_argument(
        "--use_scheduler",
        default=False,
        help="""Three options:
        1. `linear`: Use a linear schedule that inceases linearly over `--warmup_steps` to `--learning_rate` then decreases linearly for the rest of the training process.
        2. `onecycle`: Use the one cycle policy with a maximum learning rate of `--learning_rate`.
        (default: False, don't use any scheduler)
        3. `poly`: polynomial learning rate decay from `--learning_rate` to `--end_learning_rate`""",
    )
    parser.add_argument(
        "--end_learning_rate",
        default=2e-6,
        type=float,
        help="The ending learning rate when `--use_scheduler` is poly.",
    )
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )

    main_args = parser.parse_known_args()

    if main_args[0].mode == "abstractive":
        parser = AbstractiveSummarizer.add_model_specific_args(parser)
    else:
        parser = ExtractiveSummarizer.add_model_specific_args(parser)

    main_args = parser.parse_args()

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(main_args.logLevel),
    )

    # Set the `nlp` logging verbosity since its default is not INFO.
    # If the verbosity is not set back to the default for the library, an abundance
    # of output will be printed. See https://huggingface.co/docs/datasets/package_reference/logging_methods.html.
    nlp.logging.set_verbosity(nlp.logging.WARNING)

    # Train
    main(main_args)
