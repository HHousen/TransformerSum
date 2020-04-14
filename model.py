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
from rouge_score import rouge_scorer
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from pooling import Pooling
from data import SentencesProcessor, FSIterableDataset, pad_batch_collate
from convert_to_extractive import greedy_selection, combination_selection
from transformers import (
    ALL_PRETRAINED_MODEL_ARCHIVE_MAP,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.data.metrics import acc_and_f1
from transformers.activations import get_activation

from transformers.modeling_auto import MODEL_MAPPING

ALL_MODELS = tuple(ALL_PRETRAINED_MODEL_ARCHIVE_MAP)
MODEL_CLASSES = tuple(m.model_type for m in MODEL_MAPPING)

logger = logging.getLogger(__name__)


class Classifier(nn.Module):
    def __init__(
        self,
        web_hidden_size,
        linear_hidden=1536,
        first_dropout=0.1,
        last_dropout=0.1,
        activation_string="gelu",
    ):
        """nn.Module to classify sentences by reducing the hidden dimension to 1
        
        Arguments:
            web_hidden_size {int} -- The output hidden size from the word embedding model. Used as
                                     the input to the first linear layer in this nn.Module.
        
        Keyword Arguments:
            linear_hidden {int} -- The number of hidden parameters for this Classifier. (default: {1536})
            first_dropout {float} -- The value for dropout applied before any other layers. (default: {0.1})
            last_dropout {float} -- The dropout after the last linear layer. (default: {0.1})
            activation_string {str} -- A string representing an activation function in `get_activation()` (default: {"gelu"})
        """
        super(Classifier, self).__init__()
        self.dropout1 = nn.Dropout(first_dropout) if first_dropout else nn.Identity()
        self.dropout2 = nn.Dropout(last_dropout) if last_dropout else nn.Identity()
        self.linear1 = nn.Linear(web_hidden_size, linear_hidden)
        self.linear2 = nn.Linear(linear_hidden, 1)
        self.sigmoid = nn.Sigmoid()

        self.activation = (
            get_activation(activation_string) if activation_string else nn.Identity()
        )

    def forward(self, x):
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.sigmoid(x)
        sent_scores = x.squeeze(-1)
        return sent_scores


class ExtractiveSummarizer(pl.LightningModule):
    """
    A machine learning model that extractively summarizes an input text by scoring the sentences.
    Main class that handles the data loading, initial processing, training/testing/validating setup,
    and contains the actual model.
    """

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
        self.word_embedding_model.train()

        self.emd_model_frozen = False
        if hparams.num_frozen_steps > 0:
            self.emd_model_frozen = True
            self.freeze_web_model()

        if hparams.pooling_mode == "sent_rep_tokens":
            self.pooling_model = Pooling(sent_rep_tokens=True, mean_tokens=False)
        else:
            self.pooling_model = Pooling(sent_rep_tokens=False, mean_tokens=True)

        self.encoder = Classifier(self.word_embedding_model.config.hidden_size)

        # BCELoss: https://pytorch.org/docs/stable/nn.html#bceloss
        # `reduction` is "none" so the mean can be computed with padding ignored.
        # See `compute_loss()` for more info.
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

    def forward(
        self,
        input_ids,
        attention_mask,
        sent_rep_mask=None,
        token_type_ids=None,
        sent_rep_token_ids=None,
        sent_lengths=None,
        sent_lengths_mask=None,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if not self.hparams.no_use_token_type_ids:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.word_embedding_model(**inputs)
        word_vectors = outputs[0]

        sents_vec, mask = self.pooling_model(
            word_vectors=word_vectors,
            sent_rep_token_ids=sent_rep_token_ids,
            sent_rep_mask=sent_rep_mask,
            sent_lengths=sent_lengths,
            sent_lengths_mask=sent_lengths_mask,
        )

        sent_scores = self.encoder(sents_vec) * mask.float()
        return sent_scores

    def unfreeze_web_model(self):
        """ Un-freezes the `word_embedding_model` """
        for param in self.word_embedding_model.parameters():
            param.requires_grad = True

    def freeze_web_model(self):
        """ Freezes the encoder `word_embedding_model` """
        for param in self.word_embedding_model.parameters():
            param.requires_grad = False

    def compute_loss(self, outputs, labels, mask):
        # The below is the same as BCELoss(reduction="mean") except the below
        # ignores padding values in the calculation of the mean.
        # In other words, this:
        # loss_func = nn.BCELoss(reduction="mean")
        # final_loss = loss_func(outputs, labels.float())
        # is the same as this:
        # loss_func = nn.BCELoss(reduction="none")
        # loss = loss_func(outputs, labels.float())
        # total_loss = loss.sum()
        # final_loss = total_loss / loss.numel()
        # but for the below code `loss.numel()` is replaced with the number of non-padding elements
        loss = self.loss_func(outputs, labels.float())
        # set all padding values to zero
        loss = loss * mask.float()
        # add up all the loss values for each sequence (including padding because
        # padding values are zero and thus will have no effect)
        sum_loss_per_sequence = loss.sum(dim=1)
        # count the number of losses that are not padding per sequence
        num_not_padded_per_sequence = mask.sum(dim=1).float()
        # find the average loss per sequence
        average_per_sequence = sum_loss_per_sequence / num_not_padded_per_sequence
        # get the sum of the average loss per sequence
        sum_avg_seq_loss = average_per_sequence.sum()  # sum_average_per_sequence
        # get the mean of `average_per_sequence`
        batch_size = average_per_sequence.size(0)
        mean_avg_seq_loss = sum_avg_seq_loss / batch_size

        # calculate the sum of all the loss values for each sequence
        total_loss = sum_loss_per_sequence.sum()
        # count the total number of losses that are not padding
        total_num_not_padded = num_not_padded_per_sequence.sum().float()
        # average loss
        average_loss = total_loss / total_num_not_padded
        # total loss normalized by batch size
        total_norm_batch_loss = total_loss / batch_size
        return (
            total_loss,
            total_norm_batch_loss,
            sum_avg_seq_loss,
            mean_avg_seq_loss,
            average_loss,
        )

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
        all_targets = []
        for doc in documents:  # for each document in the json file
            source = doc["src"]
            if "tgt" in doc:
                target = doc["tgt"]
                all_targets.append(target)

            if oracle_mode and oracle_mode != "none":
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
                targets=all_targets if all_targets else None,
                overwrite_examples=True,
                overwrite_labels=True,
            )
        else:
            # otherwise add ids to `labels`
            processor.add_examples(
                all_sources,
                labels=all_ids,
                targets=all_targets if all_targets else None,
                overwrite_examples=True,
                overwrite_labels=True,
            )

        processor.get_features(
            tokenizer,
            bert_compatible_cls=hparams.processor_no_bert_compatible_cls,
            create_segment_ids=hparams.create_token_type_ids,
            sent_rep_token_id="cls",
            create_source=(
                True if all_targets else False
            ),  # create the source if targets were present
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
                dataset_files, batch_size=batch_size, verbose=True
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
        self.rouge_metrics = ["rouge1", "rouge2", "rougeL"]
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.rouge_metrics, use_stemmer=True
        )
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
            # set `max_epochs` if it has not been set
            if not self.hparams.max_epochs:
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
        # parameters = [
        #     {"params": self.encoder.parameters()},
        #     {
        #         "params": self.word_embedding_model.parameters(),
        #         "lr": self.hparams.web_learning_rate,
        #     },
        # ]
        if self.hparams.optimizer_type == "ranger":
            from optimizers.ranger.ranger import Ranger

            optimizer = Ranger(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                k=self.hparams.ranger_k,
                eps=self.hparams.adam_epsilon,
            )
        elif self.hparams.optimizer_type == "yellowfin":
            from optimizers.yellowfin import YFOptimizer

            optimizer = YFOptimizer(self.parameters())
        elif self.hparams.optimizer_type == "qhadam":
            from qhoptim.pyt import QHAdam

            optimizer = QHAdam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                nus=(0.7, 1.0),
                betas=(0.995, 0.999),
            )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,  # optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )

        if self.hparams.use_scheduler:
            if self.hparams.use_scheduler == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.hparams.warmup_steps,
                    num_training_steps=t_total,
                )
            elif self.hparams.use_scheduler == "onecycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.hparams.learning_rate, total_steps=t_total
                )
            else:
                logger.error(
                    "The value "
                    + str(self.hparams.use_scheduler)
                    + " for `--use_scheduler` is invalid."
                )
            scheduler_dict = {"scheduler": scheduler, "interval": "step"}
            return ([optimizer], [scheduler_dict])
        else:
            return optimizer

    def on_batch_end(self):
        """
        PyTorch Lightning hook to do the following:
        1. Log the learning_rate
        2. Begin training the `word_embedding_model` after `num_frozen_steps` steps
        """
        if self.hparams.use_scheduler:
            last_lrs = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()
            # if the logger is tensorboard then use the `add_scalar` method
            if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
                self.logger.experiment.add_scalar(
                    "learning_rate",
                    (last_lrs[0] if isinstance(last_lrs, list) else last_lrs),
                    self.trainer.global_step,
                )
            # if the logger is for wandb.ai
            elif isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                self.logger.experiment.log(
                    {
                        "learning_rate": (
                            last_lrs[0] if isinstance(last_lrs, list) else last_lrs
                        ),
                    }
                )

        if self.emd_model_frozen and (
            self.trainer.global_step > self.hparams.num_frozen_steps
        ):
            self.emd_model_frozen = False
            self.unfreeze_web_model()

    def training_step(self, batch, batch_idx):
        # Get batch information
        labels = batch["labels"]

        # delete labels so now batch contains everything to be inputted into the model
        del batch["labels"]

        # Compute model forward
        outputs = self.forward(**batch)

        # Compute loss
        (
            loss_total,
            loss_total_norm_batch,
            loss_avg_seq_sum,
            loss_avg_seq_mean,
            loss_avg,
        ) = self.compute_loss(outputs, labels, batch["sent_rep_mask"])

        # Generate logs
        tqdm_dict = {
            "train_loss_total": loss_total,
            "train_loss_total_norm_batch": loss_total_norm_batch,
            "train_loss_avg_seq_sum": loss_avg_seq_sum,
            "train_loss_avg_seq_mean": loss_avg_seq_mean,
            "train_loss_avg": loss_avg,
        }
        output = OrderedDict(
            {"loss": loss_total, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch, batch_idx):
        # Get batch information
        labels = batch["labels"]

        # delete labels so now batch contains everything to be inputted into the model
        del batch["labels"]

        # Compute model forward
        outputs = self.forward(**batch)

        # Compute loss
        (
            loss_total,
            loss_total_norm_batch,
            loss_avg_seq_sum,
            loss_avg_seq_mean,
            loss_avg,
        ) = self.compute_loss(outputs, labels, batch["sent_rep_mask"])

        # Compute accuracy metrics
        y_hat = outputs
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        y_hat = torch.flatten(y_hat)
        y_true = torch.flatten(labels)
        result = acc_and_f1(y_hat.detach().cpu().numpy(), y_true.detach().cpu().numpy())
        acc = torch.tensor(result["acc"])
        f1 = torch.tensor(result["f1"])
        acc_f1 = torch.tensor(result["acc_and_f1"])

        output = OrderedDict(
            {
                "val_loss_total": loss_total,
                "val_loss_total_norm_batch": loss_total_norm_batch,
                "val_loss_avg_seq_sum": loss_avg_seq_sum,
                "val_loss_avg_seq_mean": loss_avg_seq_mean,
                "val_loss_avg": loss_avg,
                "val_acc": acc,
                "val_f1": f1,
                "val_acc_and_f1": acc_f1,
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        # Get the average loss and accuracy metrics over all evaluation runs
        avg_loss_total = torch.stack([x["val_loss_total"] for x in outputs]).mean()
        avg_loss_total_norm_batch = torch.stack(
            [x["val_loss_total_norm_batch"] for x in outputs]
        ).mean()
        avg_loss_avg_seq_sum = torch.stack(
            [x["val_loss_avg_seq_sum"] for x in outputs]
        ).mean()
        avg_loss_avg_seq_mean = torch.stack(
            [x["val_loss_avg_seq_mean"] for x in outputs]
        ).mean()
        avg_loss_avg = torch.stack([x["val_loss_avg"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()
        avg_val_acc_and_f1 = torch.stack([x["val_acc_and_f1"] for x in outputs]).mean()

        # Generate logs
        tqdm_dict = {
            "val_loss_total": avg_loss_total,
            "val_loss_total_norm_batch": avg_loss_total_norm_batch,
            "val_loss_avg_seq_sum": avg_loss_avg_seq_sum,
            "val_loss_avg_seq_mean": avg_loss_avg_seq_mean,
            "val_loss_avg": avg_loss_avg,
            "val_acc": avg_val_acc,
            "val_f1": avg_val_f1,
            "val_acc_and_f1": avg_val_acc_and_f1,
        }
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": avg_loss_avg_seq_sum,
        }
        return result

    def test_step(self, batch, batch_idx):
        # Get batch information
        labels = batch["labels"]
        sources = batch["source"]
        targets = batch["target"]

        # delete labels, sources, and targets so now batch contains everything to be inputted into the model
        del batch["labels"]
        del batch["source"]
        del batch["target"]

        # Compute model forward
        outputs = self.forward(**batch)

        # Compute accuracy metrics
        y_hat = outputs.clone().detach()
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        y_hat = torch.flatten(y_hat)
        y_true = torch.flatten(labels)
        result = acc_and_f1(y_hat.detach().cpu().numpy(), y_true.detach().cpu().numpy())
        acc = torch.tensor(result["acc"])
        f1 = torch.tensor(result["f1"])
        acc_f1 = torch.tensor(result["acc_and_f1"])

        rouge_outputs = {}
        sorted_ids = (
            torch.argsort(outputs, dim=1, descending=True).detach().cpu().numpy()
        )
        if self.hparams.test_id_method == "top_k":
            selected_ids = sorted_ids[:, : self.hparams.test_k]
        elif self.hparams.test_id_method == "greater_k":
            # `indexes` is sorted by original sentence order (sentences that appear first in the
            # original document are first in the summary)
            # if none of the rankings for a sample are greater than `test_k` then the top 3
            # sorted by ranking are used
            indexes = np.argwhere(outputs.detach().cpu().numpy() > self.hparams.test_k)
            selected_ids = [[] for _ in range(outputs.size(0))]
            previous_index = -1
            # if the final document did not have any values greater than `hparams.test_k`
            # then set it to the -1 (the skip token checked below)
            final_index = outputs.size(0) - 1
            if indexes.size == 0 or indexes[-1, 0] != final_index:
                indexes = np.append(indexes, [[final_index, -1]], axis=0)

            for index, value in indexes:
                # if the index has changed and is not one greater then the previous then
                # index was skipped because no elements greater than k
                if (
                    (previous_index != index) and (previous_index + 1 != index)
                ) or value == -1:
                    # For the first time the above loop runs, `previous_index` is -1 because no
                    # no index has been checked yet. The -1 is necessary to check if the 0th
                    # index is skipped. But, if the 0th index is skipped then the values need to be
                    # added to the 0th index, not the -1st, so 1 is added to `previous_index` to
                    # make it 0.
                    if previous_index == -1:
                        previous_index += 1
                    # multiple entires might have been skipped
                    num_skipped = index - previous_index
                    for idx in range(num_skipped):
                        # the index was skipped so add the top three for that index
                        selected_ids[previous_index + idx] = sorted_ids[
                            previous_index + idx, :3
                        ].tolist()
                # current entry was marked as skip
                if value == -1:
                    selected_ids[index] = sorted_ids[index, :3].tolist()
                else:
                    selected_ids[index].append(value)
                previous_index = index
        else:
            logger.error(
                str(self.hparams.test_id_method)
                + " is not a valid option for `--test_id_method`."
            )
        # get ROUGE scores for each (source, target) pair
        for idx, (source, source_ids, target) in enumerate(
            zip(sources, selected_ids, targets)
        ):
            current_prediction = ""
            target = " ".join(target)
            for i in source_ids:
                candidate = source[i].strip()
                current_prediction += candidate
            result = self.rouge_scorer.score(target, current_prediction)
            for key, item in result.items():
                if key not in rouge_outputs:
                    rouge_outputs[key] = []
                rouge_outputs[key].append(item)

        output = OrderedDict(
            {"test_acc": acc, "test_f1": f1, "test_acc_and_f1": acc_f1}
        )
        output = {**output, **rouge_outputs}
        return output

    def test_epoch_end(self, outputs):
        # Get the accuracy metrics over all testing runs
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        avg_test_f1 = torch.stack([x["test_f1"] for x in outputs]).mean()
        avg_test_acc_and_f1 = torch.stack(
            [x["test_acc_and_f1"] for x in outputs]
        ).mean()
        rouge_scores_log = {}
        for metric in self.rouge_metrics:
            rouge_scores_log[metric + "-precision"] = np.mean(
                [y.precision for batch_list in outputs for y in batch_list[metric]]
            )
            rouge_scores_log[metric + "-recall"] = np.mean(
                [y.recall for batch_list in outputs for y in batch_list[metric]]
            )
            rouge_scores_log[metric + "-fmeasure"] = np.mean(
                [y.fmeasure for batch_list in outputs for y in batch_list[metric]]
            )

        # Generate logs
        tqdm_dict = {
            "test_acc": avg_test_acc,
            "test_f1": avg_test_f1,
            "avg_test_acc_and_f1": avg_test_acc_and_f1,
        }
        log = {**tqdm_dict, **rouge_scores_log}
        result = {"progress_bar": tqdm_dict, "log": log}
        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="bert-base-uncased",
            help="Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(ALL_MODELS),
        )
        parser.add_argument(
            "--model_type",
            type=str,
            default="bert",
            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
        )
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
        parser.add_argument("--weight_decay", default=1e-2, type=float)
        parser.add_argument(
            "--pooling_mode",
            type=str,
            default="sent_rep_tokens",
            choices=["sent_rep_tokens", "mean_tokens"],
            help="How word vectors should be converted to sentence embeddings.",
        )
        parser.add_argument(
            "--web_learning_rate",
            default=1e-05,
            type=float,
            help="Word embedding model specific learning rate.",
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
            help="""Which optimizer to use:
            1. `ranger` optimizer (combination of RAdam and LookAhead)
            2. `adamw`
            3. `yellowfin`""",
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
            help="Linear warmup over warmup_steps. Only active if `--use_scheduler` is set.",
        )
        parser.add_argument(
            "--use_scheduler",
            default=False,
            help="""Two options:
            1. `linear`: Use a linear schedule that inceases linearly over `--warmup_steps` to `--learning_rate` then decreases linearly for the rest of the training process.
            2. `onecycle`: Use the one cycle policy with a maximum learning rate of `--learning_rate`.
            (default: False, don't use any scheduler)""",
        )
        parser.add_argument(
            "--num_frozen_steps",
            type=int,
            default=0,
            help="Freeze (don't train) the word embedding model for this many steps.",
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
            help="Create token type ids during preprocessing.",
        )
        parser.add_argument(
            "--no_use_token_type_ids",
            action="store_true",
            help="Set to not train with `token_type_ids` (don't pass them into the model).",
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
        parser.add_argument(
            "--test_id_method",
            type=str,
            default="greater_k",
            choices=["greater_k", "top_k"],
            help="How to chose the top predictions from the model for ROUGE scores.",
        )
        parser.add_argument(
            "--test_k",
            type=float,
            default=0.5,
            help="The `k` parameter for the `--test_id_method`. Must be set if using `top_k` option. (default: 0.5)",
        )
        return parser
