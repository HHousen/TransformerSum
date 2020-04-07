import logging
from pytorch_lightning import Trainer
from model import ExtractiveSummarizer
from argparse import ArgumentParser


def main(args):
    model = ExtractiveSummarizer(hparams=args)

    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # parametrize the network: general options
    parser.add_argument(
        "--default_save_path",
        required=True,
        type=str,
        help="Default path for logs and weights",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
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
