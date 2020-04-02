from pytorch_lightning import Trainer
from model import ExtractiveSummarizer
from argparse import ArgumentParser

def main(args):
    model = ExtractiveSummarizer(hparams=args)

    # makes all trainer options available from the command line
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()

    # adds all the trainer options as default arguments (like max_epochs)
    parser = Trainer.add_argparse_args(parser)

    # parametrize the network: general options
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    args = parser.parse_args()

    # train
    main(args)
