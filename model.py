# 1. Compute regular embeddings
# 2. Compute sentence embeddings
# 3. Run through linear layer

import os
import sys
import glob
import logging
import json
import gzip
import numpy as np
from functools import partial
from multiprocessing import Pool
from collections import OrderedDict
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from Pooling import Pooling
from data import SentencesProcessor, FSIterableDataset, pad_batch_collate
from convert_to_extractive import greedy_selection, combination_selection
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.data.metrics import acc_and_f1

logger = logging.getLogger(__name__)


class Classifier(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob=0.2):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        sent_scores = self.sigmoid(x)
        return sent_scores


class ExtractiveSummarizer(pl.LightningModule):
    def __init__(self, hparams, embedding_model_config=None):
        super(ExtractiveSummarizer, self).__init__()

        self.hparams = hparams

        if not embedding_model_config:
            embedding_model_config = AutoConfig.from_pretrained(
                hparams.model_name_or_path
            )
        self.word_embedding_model = AutoModel.from_pretrained(
            hparams.model_name_or_path, config=embedding_model_config
        )

        self.pooling_model = Pooling(
            self.word_embedding_model.config.hidden_size,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        self.encoder = Classifier(self.word_embedding_model.config.hidden_size)

        self.loss_func = nn.BCELoss(reduction="none")

        # Data
        self.processor = SentencesProcessor(name="main_processor")

        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name
            if hparams.tokenizer_name
            else hparams.model_name_or_path,
            do_lower_case=hparams.tokenizer_lowercase,
        )
        self.train_dataloader_object = None  # not created yet

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
        parser.add_argument("--model_type", type=str, default="bert")
        parser.add_argument("--tokenizer_name", type=str, default="")
        parser.add_argument("--tokenizer_lowercase", action="store_true")
        parser.add_argument("--max_seq_length", type=int, default=512)
        parser.add_argument(
            "--oracle_mode",
            type=str,
            choices=["none", "greedy", "combination"],
            default="none",
        )
        parser.add_argument("--data_path", type=str, required=True)
        parser.add_argument("--num_threads", type=int, default=4)
        parser.add_argument("--processing_num_threads", type=int, default=2)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps.",
        )
        parser.add_argument(
            "--train_batch_size",
            default=8,
            type=int,
            help="Batch size per GPU/CPU for training.",
        )
        parser.add_argument(
            "--val_batch_size",
            default=8,
            type=int,
            help="Batch size per GPU/CPU for evaluation.",
        )
        parser.add_argument(
            "--test_batch_size",
            default=8,
            type=int,
            help="Batch size per GPU/CPU for testing.",
        )
        parser.add_argument(
            "--processor_no_bert_compatible_cls",
            action="store_false",
            help="If model uses bert compatible [CLS] tokens for sentence representations.",
        )
        parser.add_argument(
            "--only_preprocess",
            action="store_true",
            help="""Only preprocess and write the data to disk. Don't train model. 
            This will force data to be preprocessed, even if it was already computed and 
            is detected on disk, and any previous processed files will be overwritten.""",
        )
        parser.add_argument(
            "--create_token_type_ids",
            type=str,
            choices=["binary", "sequential"],
            default="binary",
            help="Create token type ids.",
        )
        parser.add_argument(
            "--train_name",
            type=str,
            default="train",
            help="name for set of training files on disk (for loading and saving)",
        )
        parser.add_argument(
            "--val_name",
            type=str,
            default="val",
            help="name for set of validation files on disk (for loading and saving)",
        )
        parser.add_argument(
            "--test_name",
            type=str,
            default="test",
            help="name for set of testing files on disk (for loading and saving)",
        )
        # parser.add_argument(
        #     "--data_splits",
        #     default=["train", "valid", "test"],
        #     choices=["train", "valid", "test"],
        #     nargs="+",
        #     help="which dataset splits to process",
        # )
        parser.add_argument("--load_data_from_disk", action="store_true")
        return parser

    def forward(
        self,
        input_ids,
        attention_mask,
        sent_rep_mask=None,
        token_type_ids=None,
        sent_rep_token_ids=None,
    ):
        word_vectors, pooler_output = self.word_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sents_vec = word_vectors[
            torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids
        ].squeeze()
        sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
        # reduces the [batch_size, num_sents, hidden_size] hidden_size to 1
        sent_scores = (
            self.encoder(sents_vec).squeeze(-1) * sent_rep_mask.float()
        )  # multiply by `sent_rep_mask` to only include non-padded tokens

        # print(sent_rep_token_ids)
        # print(word_vectors.shape)
        # print(sents_vec.shape)
        # print(sent_scores.shape)

        return sent_scores

    def compute_loss(self, outputs, labels, mask):
        loss = self.loss_func(outputs, labels.float())
        loss = (loss * mask.float()).sum()
        return loss / loss.numel()

    def json_to_dataset(
        self,
        tokenizer,
        hparams,
        inputs=None,
        num_files=0,
        processor=None,
        oracle_mode=None,
    ):
        idx, json_file = inputs
        logger.info(
            "Processing "
            + str(json_file)
            + " ("
            + str(idx + 1)  # because starts at 0 but num_files starts at 1
            + "/"
            + str(num_files)
            + ")"
        )

        # open current json file (which is a set of documents)
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

        all_sources = []
        all_ids = []
        for doc in documents:  # for each document in the json file
            source = doc["src"]
            if oracle_mode and oracle_mode != "none":
                tgt = doc["tgt"]
                # source and tgt are now arrays of sentences where each sentence is an array of tokens
                if oracle_mode == "greedy":
                    ids = greedy_selection(source, tgt, 3)
                elif oracle_mode == "combination":
                    ids = combination_selection(source, tgt, 3)
            else:  # default is not to extract oracle ids
                ids = doc["labels"]

            all_sources.append(source)
            all_ids.append(ids)

        if oracle_mode and oracle_mode != "none":
            # if using `oracle_mode` then add ids to `oracle_ids`
            processor.add_examples(
                all_sources,
                oracle_ids=all_ids,
                overwrite_examples=True,
                overwrite_labels=True,
            )
        else:
            # otherwise add ids to `labels`
            processor.add_examples(
                all_sources,
                labels=all_ids,
                overwrite_examples=True,
                overwrite_labels=True,
            )

        processor.get_features(
            tokenizer,
            bert_compatible_cls=hparams.processor_no_bert_compatible_cls,
            create_segment_ids=hparams.create_token_type_ids,
            n_process=hparams.processing_num_threads,
            max_length=hparams.max_seq_length,
            pad_on_left=bool(
                hparams.model_type in ["xlnet"]
            ),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            return_type="lists",
            save_to_path=hparams.data_path,
            save_to_name=os.path.basename(file_path),
        )

    def prepare_data(self):
        datasets = dict()

        # loop through all data_splits
        data_splits = [
            self.hparams.train_name,
            self.hparams.val_name,
            self.hparams.test_name,
        ]
        # save batch sizes in same order
        batch_sizes = [
            self.hparams.train_batch_size,
            self.hparams.val_batch_size,
            self.hparams.test_batch_size,
        ]
        for corpus_type, batch_size in zip(data_splits, batch_sizes):
            # get the current list of dataset files. if preprocessing has already happened
            # then this will be the list of files that should be passed to a FSIterableDataset.
            # if preprocessing has not happened then `dataset_files` should be an empty list
            # and the data will be processed
            dataset_files = glob.glob(
                os.path.join(self.hparams.data_path, "*" + corpus_type + ".*.pt")
            )
            # if no dataset files detected or model is set to `only_preprocess`
            if (not dataset_files) or (self.hparams.only_preprocess):
                json_files = glob.glob(
                    os.path.join(self.hparams.data_path, "*" + corpus_type + ".*.json*")
                )

                num_files = len(json_files)

                # pool = Pool(self.hparams.num_threads)
                json_to_dataset_processor = partial(
                    self.json_to_dataset,
                    self.tokenizer,
                    self.hparams,
                    num_files=num_files,
                    processor=self.processor,
                    oracle_mode=self.hparams.oracle_mode,
                )

                for result in map(
                    json_to_dataset_processor, zip(range(len(json_files)), json_files),
                ):
                    pass
                # pool.close()
                # pool.join()

            # if set to only preprocess the data then continue to next loop (aka next split of dataset)
            if self.hparams.only_preprocess:
                continue

            # always create actual dataset, either after writing the shard ".pt" files to disk
            # or by skipping that step (because preprocessed ".pt" files detected) and going right to loading.
            # `FSIterableDataset` needs to know `batch_size` in order to properly tell the DataLoader
            # how many steps there are per epoch. Since `FSIterableDataset` is an `IterableDataset` the
            # `DataLoader` will ask the `Dataset` for the length instead of calculating it because
            # the length of `IterableDatasets` might not be known, but it is in this case.
            datasets[corpus_type] = FSIterableDataset(
                dataset_files, batch_size=batch_size
            )

        # if set to only preprocess the data then exit after all loops have been completed
        if self.hparams.only_preprocess:
            logger.warn(
                "Exiting since data has been preprocessed and written to disk and `hparams.only_preprocess` is True."
            )
            sys.exit(0)

        self.datasets = datasets

    def train_dataloader(self):
        if self.train_dataloader_object:
            return self.train_dataloader_object

        train_dataset = self.datasets[self.hparams.train_name]
        # train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            # sampler=train_sampler,
            batch_size=self.hparams.train_batch_size,
            collate_fn=pad_batch_collate,
        )

        self.train_dataloader_object = train_dataloader
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = self.datasets[self.hparams.val_name]
        # valid_sampler = RandomSampler(valid_dataset)
        valid_dataloader = DataLoader(
            valid_dataset,
            # sampler=valid_sampler,
            batch_size=self.hparams.val_batch_size,
            collate_fn=pad_batch_collate,
        )
        return valid_dataloader

    def test_dataloader(self):
        test_dataset = self.datasets[self.hparams.test_name]
        # test_sampler = RandomSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            # sampler=test_sampler,
            batch_size=self.hparams.test_batch_size,
            collate_fn=pad_batch_collate,
        )
        return test_dataloader

    def configure_optimizers(self):
        self.train_dataloader_object = (
            self.train_dataloader()
        )  # create the train dataloader so the number of examples can be determined
        if (
            self.hparams.max_steps and self.hparams.max_steps > 0
        ):  # check that max_steps is not None and is greater than 0
            t_total = self.hparams.max_steps
            self.hparams.max_epochs = (
                self.hparams.max_steps
                // (
                    len(self.train_dataloader_object)
                    // self.hparams.accumulate_grad_batches
                )
                + 1
            )
        else:
            t_total = (
                len(self.train_dataloader_object)
                // self.hparams.accumulate_grad_batches
                * self.hparams.max_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        return ([optimizer], [scheduler])

    def training_step(self, batch, batch_idx):
        # Get batch information
        labels = batch["labels"]

        # delete labels so now batch contains everything to be inputted into the model
        del batch["labels"]

        # inputs = {
        #     "input_ids": batch["input_ids"],
        #     "attention_mask": batch["attention_mask"],
        #     "token_type_ids": batch["token_type_ids"],
        #     "sent_rep_token_ids": batch["sent_rep_token_ids"],
        #     "sent_rep_mask": batch["sent_rep_mask"],
        # }
        # # Get batch information (indices specified in SentencesProcessor.get_features())
        # labels = batch[2]
        # inputs = {
        #     "input_ids": batch[0],
        #     "attention_mask": batch[1],
        #     "token_type_ids": batch[3],
        #     "sent_rep_token_ids": batch[4],
        #     "sent_rep_mask": batch[5],
        # }

        # Compute model forward
        outputs = self.forward(**batch)

        # Compute loss
        loss = self.compute_loss(outputs, labels, batch["sent_rep_mask"])

        # Generate logs
        tqdm_dict = {"train_loss": loss}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch, batch_idx):
        # Get batch information
        labels = batch["labels"]

        # delete labels so now batch contains everything to be inputted into the model
        del batch["labels"]
        # inputs = {
        #     "input_ids": batch["input_ids"],
        #     "attention_mask": batch["attention_mask"],
        #     "token_type_ids": batch["token_type_ids"],
        #     "sent_rep_token_ids": batch["sent_rep_token_ids"],
        #     "sent_rep_mask": batch["sent_rep_mask"],
        # }

        # Compute model forward
        outputs = self.forward(**batch)

        # Compute loss
        loss = self.compute_loss(outputs, labels, batch["sent_rep_mask"])

        # Compute accuracy metrics
        y_hat = outputs
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        y_hat = torch.flatten(y_hat)
        result = acc_and_f1(
            y_hat.detach().cpu().numpy(), labels.detach().cpu().numpy().flatten()
        )
        acc = torch.tensor(result["acc"])
        f1 = torch.tensor(result["f1"])
        acc_f1 = torch.tensor(result["acc_and_f1"])

        output = OrderedDict(
            {"val_loss": loss, "val_acc": acc, "val_f1": f1, "val_acc_and_f1": acc_f1}
        )
        return output

    def validation_epoch_end(self, outputs):
        # Get the average loss and accuracy metrics over all evaluation runs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()
        avg_val_acc_and_f1 = torch.stack([x["val_acc_and_f1"] for x in outputs]).mean()

        # Generate logs
        tqdm_dict = {
            "val_loss": avg_loss,
            "val_acc": avg_val_acc,
            "val_f1": avg_val_f1,
            "avg_val_acc_and_f1": avg_val_acc_and_f1,
        }
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": avg_loss,
        }
        return result

    def test_step(self, batch, batch_idx):
        # Get batch information
        labels = batch['labels']

        # delete labels so now batch contains everything to be inputted into the model
        del batch["labels"]
        
        # Compute model forward
        outputs = self.forward(**batch)

        # Compute accuracy metrics
        result = acc_and_f1(
            outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()
        )
        acc = torch.tensor(result["acc"])
        f1 = torch.tensor(result["f1"])
        acc_f1 = torch.tensor(result["acc_and_f1"])

        output = OrderedDict(
            {"test_acc": acc, "test_f1": f1, "test_acc_and_f1": acc_f1}
        )
        return output

    def test_epoch_end(self, outputs):
        # Get the accuracy metrics over all testing runs
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        avg_test_f1 = torch.stack([x["test_f1"] for x in outputs]).mean()
        avg_test_acc_and_f1 = torch.stack(
            [x["test_acc_and_f1"] for x in outputs]
        ).mean()

        # Generate logs
        tqdm_dict = {
            "test_acc": avg_test_acc,
            "test_f1": avg_test_f1,
            "avg_test_acc_and_f1": avg_test_acc_and_f1,
        }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict}
        return result
