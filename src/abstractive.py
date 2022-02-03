import itertools
import logging
import os
import random
import sys
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
from time import time

import numpy as np
import pyarrow
import pytorch_lightning as pl
import spacy
import torch
from rouge_score import rouge_scorer, scoring
from spacy.lang.en import English
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, EncoderDecoderModel

import datasets as nlp
from convert_to_extractive import tokenize
from helpers import (
    LabelSmoothingLoss,
    SortishSampler,
    generic_configure_optimizers,
    pad,
    pad_tensors,
    test_rouge,
)

logger = logging.getLogger(__name__)


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by ``pad_token_id``."""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)

    if attention_mask is None:
        return input_ids[:, keep_column_mask]

    return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def longformer_modifier(final_dictionary, tokenizer, attention_window):
    """
    Creates the `global_attention_mask` for the longformer. Tokens with global attention
    attend to all other tokens, and all other tokens attend to them. This is important for
    task-specific finetuning because it makes the model more flexible at representing the
    task. For example, for classification, the `<s>` token should be given global attention.
    For QA, all question tokens should also have global attention. For summarization,
    global attention is given to all of the `<s>` (RoBERTa 'CLS' equivalent) tokens. Please
    refer to the `Longformer paper <https://arxiv.org/abs/2004.05150>`_ for more details. Mask
    values selected in ``[0, 1]``: ``0`` for local attention, ``1`` for global attention.
    """
    # `batch_size` is the number of attention masks (one mask per input sequence)
    batch_size = len(final_dictionary["source_mask"])
    # `sequence_length` is the number of tokens for the first sequence in the batch
    sequence_length = len(final_dictionary["source_mask"][0])
    # create `global_attention_mask` using the above details
    global_attention_mask = torch.tensor([[0] * sequence_length] * batch_size)
    # set the `sent_rep_token_ids` to 1, which is global attention
    for idx, input_sequence in enumerate(final_dictionary["source"]):
        for inner_idx, token_id in enumerate(input_sequence):
            if token_id == tokenizer.cls_token_id:
                global_attention_mask[idx, inner_idx] = 1

    final_dictionary["global_attention_mask"] = global_attention_mask

    for key, item in final_dictionary.items():
        final_dictionary[key] = pad_tensors(
            item,
            nearest_multiple_of=attention_window[0],
        )

    return final_dictionary


class AbstractiveSummarizer(pl.LightningModule):
    """
    A machine learning model that abstractively summarizes an input text using a seq2seq model.
    Main class that handles the data loading, initial processing, training/testing/validating setup,
    and contains the actual model.
    """

    def __init__(self, hparams):
        super(AbstractiveSummarizer, self).__init__()

        self.save_hyperparameters(hparams)

        if len(self.hparams.dataset) <= 1:
            self.hparams.dataset = self.hparams.dataset[0]

        if self.hparams.decoder_model_name_or_path:
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.hparams.model_name_or_path,
                (
                    self.hparams.decoder_model_name_or_path
                    if self.hparams.decoder_model_name_or_path
                    else self.hparams.model_name_or_path
                ),
                gradient_checkpointing=self.hparams.gradient_checkpointing,
                tie_encoder_decoder=self.hparams.tie_encoder_decoder,
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.hparams.model_name_or_path,
                gradient_checkpointing=self.hparams.gradient_checkpointing,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path, use_fast=True
        )

        if self.hparams.model_max_length:
            self.tokenizer.model_max_length = self.hparams.model_max_length

        self.rouge_sentence_split_token = "<q>"
        self.tokenizer.add_tokens(self.rouge_sentence_split_token)
        self.rouge_sentence_split_token_id = self.tokenizer.convert_tokens_to_ids(
            self.rouge_sentence_split_token
        )

        # bo = beginning of
        # eo = ending of
        # seq = sequence (not using 's' because 's' stands for sentence in other places)
        # Use `bos_token` for boseq if `bos_token` is set, otherwise use "[unused0]"
        # Use `pad_token` for eoseq if `pad_token` is set, otherwise use "[unused1]"
        do_seq_special_add = False
        if self.tokenizer.bos_token:
            self.target_boseq_token = self.tokenizer.bos_token
        else:
            self.target_boseq_token = "[unused0]"
            do_seq_special_add = True

        if self.tokenizer.pad_token:
            self.target_eoseq_token = self.tokenizer.pad_token
        else:
            self.target_eoseq_token = "[unused1]"
            self.tokenizer.pad_token = "[unused2]"
            do_seq_special_add = True

        # Convert `target_boseq_token` and `target_eoseq_token` to IDs
        self.target_boseq_token_id = self.tokenizer.convert_tokens_to_ids(
            self.target_boseq_token
        )
        self.target_eoseq_token_id = self.tokenizer.convert_tokens_to_ids(
            self.target_eoseq_token
        )

        # If the `*oseq` tokens are not already "special" then add them as special
        # tokens so that they are ignored when decoding.
        if do_seq_special_add:
            special_tokens_dict = {
                "additional_special_tokens": [
                    self.target_boseq_token,
                    self.target_eoseq_token,
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.hparams.label_smoothing > 0:
            self.loss_func = LabelSmoothingLoss(
                self.hparams.label_smoothing,
                self.tokenizer.vocab_size,
                ignore_index=self.tokenizer.pad_token_id,
            )
        else:
            self.loss_func = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id
            )

        self.train_dataloader_object = None  # not created yet
        self.rouge_metrics = None
        self.rouge_scorer = None
        self.dataset = {}

        self.tokenized_data_file_paths = {}
        for split in ["train", "validation", "test"]:
            features_cache_file = os.path.join(
                self.hparams.cache_file_path, (split + "_tokenized")
            )
            self.tokenized_data_file_paths[split] = features_cache_file

        if any(
            x in self.hparams.model_name_or_path
            for x in ["longformer", "led-base", "led-large"]
        ):
            longformer_modifier_ = partial(
                longformer_modifier,
                tokenizer=self.tokenizer,
                attention_window=self.model.config.attention_window,
            )
            self.collate_fn = partial(
                self.abs_collate_fn, modifier=longformer_modifier_
            )
        else:
            self.collate_fn = self.abs_collate_fn

    def forward(
        self,
        source=None,
        target=None,
        source_mask=None,
        target_mask=None,
        labels=None,
        **kwargs
    ):
        """Model forward function. See the `60 minute bliz tutorial <https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html>`_
        if you are unsure what a forward function is.

        Args:
            source (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, optional):
                Indices of input sequence tokens in the vocabulary for the encoder.
                `What are input IDs? <https://huggingface.co/transformers/glossary.html#input-ids>`_
                Defaults to None.
            target (``torch.LongTensor`` of shape ``(batch_size, target_sequence_length)``, optional): Provide
                for sequence to sequence training to the decoder. Defaults to None.
            source_mask (``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``, optional): Mask
                to avoid performing attention on padding token indices for the encoder. Mask values
                selected in ``[0, 1]``: ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                Defaults to None.
            target_mask (``torch.BoolTensor`` of shape ``(batch_size, tgt_seq_len)``, optional): ``source_mask``
                but for the target sequence. Is an attention mask. Defaults to None.
            labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, optional): Labels
                for computing the masked language modeling loss for the decoder. Indices should be in
                ``[-100, 0, ..., config.vocab_size]``. Tokens with indices set to ``-100`` are
                ignored (masked), the loss is only computed for the tokens with labels in
                ``[0, ..., config.vocab_size]`` Defaults to None.

        Returns:
            tuple: (cross_entropy_loss, prediction_scores) The cross entropy loss and the
            prediction scores, which are the scores for each token in the vocabulary for each
            token in the output.
        """  # noqa: E501
        # `self.model.forward()` returns `decoder_outputs + encoder_outputs` where
        # `decoder_outputs` and `encoder_outputs` are dictionaries.
        # `labels` is None here so that `huggingface/transformers` does not calculate loss
        outputs = self.model.forward(
            input_ids=source.contiguous(),
            attention_mask=source_mask,
            decoder_input_ids=target,
            decoder_attention_mask=target_mask,
            use_cache=(labels is None),
            labels=None,
            **kwargs
        )

        prediction_scores = outputs[0]

        if labels is not None:
            loss = self.calculate_loss(prediction_scores, labels)
            return loss, prediction_scores

        return prediction_scores

    def setup(self, stage):
        """
        Load the data created by :meth:`~abstractive.AbstractiveSummarizer.prepare_data`.
        The downloading and loading is broken into two functions since prepare_data is
        only called from global_rank=0, and thus is not suitable for state (self.something)
        assignment.
        """
        columns = ["source", "target", "source_mask", "target_mask"]
        if stage == "fit":
            train = nlp.Dataset.from_file(self.tokenized_data_file_paths["train"])
            validation = nlp.Dataset.from_file(
                self.tokenized_data_file_paths["validation"]
            )

            train.set_format(type="torch", columns=columns)
            validation.set_format(type="torch", columns=columns)
            self.dataset["train"] = train
            self.dataset["validation"] = validation

        if stage == "test":
            test = nlp.Dataset.from_file(self.tokenized_data_file_paths["test"])
            test.set_format(type="torch", columns=columns)
            self.dataset["test"] = test

    def prepare_data(self):
        """
        Create the data using the ``huggingface/nlp`` library. This function handles
        downloading, preprocessing, tokenization, and feature extraction.
        """
        all_tokenized_files_present = all(
            os.path.isfile(path) for path in self.tokenized_data_file_paths.values()
        )
        if self.hparams.no_prepare_data or all_tokenized_files_present:
            logger.info(
                "Skipping data preparation because `--no_prepare_data` was specified or all the "
                + "final tokenized data files are present."
            )
            if self.hparams.only_preprocess:
                logger.info(
                    "Exiting because both `--no_prepare_data` and `--only_preprocess` set."
                )
                sys.exit(0)
            return

        def convert_to_features(example_batch):
            max_length = self.tokenizer.model_max_length

            articles = example_batch[self.hparams.data_example_column]

            articles_encoded_step = []
            for idx, article in enumerate(articles):
                article = article.strip()
                try:
                    article_encoded = self.tokenizer(
                        article,
                        padding="max_length",
                        truncation=True,
                    )
                    articles_encoded_step.append(article_encoded)
                except Exception:  # skipcq: FLK-E722
                    print("Failed to tokenize article: {}".format(article))
                    sys.exit(1)

                if idx != 0:
                    current_length = len(article_encoded["input_ids"])
                    first_length = len(articles_encoded_step[0]["input_ids"])
                    assert (
                        current_length == first_length
                    ), "The length of the current input, {}, does not match the length of the first input, {}.".format(  # noqa: E501
                        current_length, first_length
                    )

            articles_encoded = {
                "input_ids": [i["input_ids"] for i in articles_encoded_step],
                "attention_mask": [i["attention_mask"] for i in articles_encoded_step],
            }

            # articles_encoded = self.tokenizer.batch_encode_plus(
            #     articles, pad_to_max_length=True, truncation=True,
            # )

            highlights = example_batch[self.hparams.data_summarized_column]

            # Tokenize highlights using spacy to split them into sentences if they were not
            # already split in the dataset (use `hparams.split_char` to specify the sentence
            # boundary character)
            if not self.hparams.split_char:
                highlights = tokenize(spacy_nlp, highlights, disable_progress_bar=True)

            sep_token = self.tokenizer.sep_token
            highlights_input_ids = []
            highlights_attention_masks = []

            # For each ground-truth summary
            for highlight in highlights:
                if self.hparams.split_char:
                    # simply split into sentences if `hparams.split_char` is specified
                    sents = highlight.split(self.hparams.split_char)
                else:
                    # `highlight` is a list of sentences where each sentence is a list of tokens
                    # Combine those tokens to create a list of sentences.
                    sents = [" ".join(list_of_ids) for list_of_ids in highlight]

                assert type(sents) is list
                assert len(sents) > 0

                # Tokenize each sentence and append the `sep_token`
                sents_tokenized = []
                for sent in sents:
                    assert type(sent) is str
                    assert len(sent) > 0
                    sent = self.tokenizer.tokenize(sent)
                    sent.append(sep_token)
                    sents_tokenized.append(sent)

                # Delete the last `sep_token` from the last sentence
                assert type(sents_tokenized[-1][-1]) is str
                del sents_tokenized[-1][-1]
                # Flatten `sents_tokenized` (a list of sentences where each sentence is a list
                # of tokens) to a list of tokens
                sents_tokenized_flat = list(
                    itertools.chain.from_iterable(sents_tokenized)
                )
                assert type(sents_tokenized_flat[0]) is str
                assert len(sents_tokenized_flat) > 0

                # Convert the tokens to `input_ids`
                # `max_length` is the max length minus 2 because we need to add the
                # beginning and ending tokens to the target
                sents_input_ids = self.tokenizer.encode_plus(
                    sents_tokenized_flat,
                    truncation=True,
                    is_split_into_words=True,
                    add_special_tokens=False,
                    max_length=(max_length - 2),
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]

                # Insert beginning of sequence token and append end of sequence token.
                sents_input_ids.insert(0, self.target_boseq_token_id)
                sents_input_ids.append(self.target_eoseq_token_id)

                # Create attention mask
                attention_mask = [1] * len(sents_input_ids)

                # Append the `input_ids` and `attention_mask`
                highlights_input_ids.append(sents_input_ids)
                highlights_attention_masks.append(attention_mask)

            # Pad the highlight input ids and attention masks to `tokenizer.max_len`.
            # The articles have already been padded because they do not need the extra
            # `boseq` and `eoseq` tokens.
            highlights_input_ids = pad(
                highlights_input_ids,
                self.tokenizer.pad_token_id,
                width=max_length,
            )
            highlights_attention_masks = pad(
                highlights_attention_masks, 0, width=max_length
            )

            return {
                "source": articles_encoded["input_ids"],
                "target": highlights_input_ids,
                "source_mask": articles_encoded["attention_mask"],
                "target_mask": highlights_attention_masks,
            }

        def remove_empty(batch_item):
            article = batch_item[self.hparams.data_example_column]
            article = article.strip()
            highlight = batch_item[self.hparams.data_summarized_column]
            highlight = highlight.strip()
            # keep_article = article and article != "\n" and article != ""
            # keep_highlight = highlight and highlight != "\n" and highlight != ""
            if self.hparams.use_percentage_of_data:
                keep_example = (
                    article
                    and highlight
                    and random.random() < self.hparams.use_percentage_of_data
                )
            else:
                keep_example = bool(article and highlight)

            return keep_example

        # Load spacy if the summary column does not contain separated sentences
        if not self.hparams.split_char:
            # load spacy english small model with the "tagger" and "ner" disabled since
            # we only need the "tokenizer" and "parser"
            # more info: https://spacy.io/usage/processing-pipelines
            if self.hparams.sentencizer:
                spacy_nlp = English()
                sentencizer = spacy_nlp.create_pipe("sentencizer")
                spacy_nlp.add_pipe(sentencizer)
            else:
                spacy_nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])

        # Combine the two sections of `scientific_papers` if it is chosen as the dataset
        if self.hparams.dataset == "scientific_papers":
            self.hparams.data_example_column = "article"
            self.hparams.data_summarized_column = "abstract"

            dataset_pubmed = nlp.load_dataset(
                "scientific_papers", "pubmed", cache_dir=self.hparams.nlp_cache_dir
            )
            dataset_arxiv = nlp.load_dataset(
                "scientific_papers", "arxiv", cache_dir=self.hparams.nlp_cache_dir
            )

            combined_dataset = {}
            for (
                split,
                save_path_final_tokenized,
            ) in self.tokenized_data_file_paths.items():
                save_path = os.path.join(
                    self.hparams.cache_file_path,
                    ("arxiv_pubmed_combined_" + split + ".arrow"),
                )
                # If the file has not been saved to disk then combine arXiv and PubMed
                # and write to file. Don't process if the final tokenized version is
                # present and can be loaded.
                if (not os.path.exists(save_path)) and (
                    not os.path.exists(save_path_final_tokenized)
                ):
                    logger.info("Joining split %s", split)
                    new = pyarrow.concat_tables(
                        [dataset_pubmed[split].data, dataset_arxiv[split].data]
                    )

                    writer = nlp.arrow_writer.ArrowWriter(path=save_path)
                    writer.write_table(new)
                else:
                    logger.info(
                        "Skipping joining split %s because it already exists", split
                    )

                if not os.path.exists(save_path_final_tokenized):
                    # Load combined dataset from file if the final tokenized version
                    # does not exist.
                    logger.info("Loading split %s", save_path)
                    combined_dataset[split] = nlp.Dataset.from_file(save_path)
                else:
                    # If the tokenzed split already exists then just store the pubmed
                    # section as a placeholder so `nlp` does not complain.
                    logger.info(
                        "NOT loading split %s because the final tokenized version already exists.",
                        save_path,
                    )
                    combined_dataset[split] = dataset_pubmed[split]

            self.dataset = combined_dataset

        else:
            if type(self.hparams.dataset) is list and "/" in self.hparams.dataset[0]:
                for (split, _), dataset_path in zip(
                    self.tokenized_data_file_paths.items(), self.hparams.dataset
                ):
                    self.dataset[split] = nlp.Dataset.from_file(dataset_path)
            else:
                self.dataset = nlp.load_dataset(
                    self.hparams.dataset,
                    self.hparams.dataset_version,
                    cache_dir=self.hparams.nlp_cache_dir,
                )

        for split, features_cache_file in self.tokenized_data_file_paths.items():
            # If the tokenized version has not been created yet, then do the initial
            # filtering so it can be created
            if not os.path.isfile(features_cache_file):
                logger.info("Removing empty examples from %s dataset", split)
                start_num_examples = len(self.dataset[split])
                self.dataset[split] = self.dataset[split].filter(
                    remove_empty,
                    cache_file_name=os.path.join(
                        self.hparams.cache_file_path, (split + "_filtered")
                    ),
                )
                end_num_examples = len(self.dataset[split])
                logger.info(
                    "Removed %i (%.2f%%) examples from the dataset.",
                    start_num_examples - end_num_examples,
                    (1 - end_num_examples / start_num_examples) * 100,
                )

            logger.info("Converting %s dataset to features", split)
            self.dataset[split] = self.dataset[split].map(
                convert_to_features,
                batched=True,
                remove_columns=self.dataset[split].data.column_names,
                cache_file_name=features_cache_file,
            )

        # Exit if set to only preprocess the data
        if self.hparams.only_preprocess:
            logger.info(
                "Exiting because data has been pre-processed and the `--only_preprocess` option "
                + "is enabled."
            )
            sys.exit(0)

    def abs_collate_fn(self, batch, modifier=None):
        pad_token_id = self.tokenizer.pad_token_id

        source_ids = torch.stack([x["source"] for x in batch])
        source_mask = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target"] for x in batch])
        target_mask = torch.stack([x["target_mask"] for x in batch])

        source_ids_trimmed, source_mask_trimmed = trim_batch(
            source_ids, pad_token_id, attention_mask=source_mask
        )
        target_ids_trimmed, target_mask_trimmed = trim_batch(
            target_ids, pad_token_id, attention_mask=target_mask
        )
        batch = {
            "source": source_ids_trimmed,
            "source_mask": source_mask_trimmed,
            "target": target_ids_trimmed,
            "target_mask": target_mask_trimmed,
        }

        if modifier:
            batch = modifier(batch)

        return batch

    def train_dataloader(self):
        """Create dataloader for training."""
        train_dataset = self.dataset["train"]

        sampler = None
        shuffle = True
        if self.hparams.sortish_sampler:
            # https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/examples/seq2seq/finetune.py#L206
            assert self.hparams.gpus <= 1
            sampler = SortishSampler(
                train_dataset,
                self.hparams.batch_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            shuffle = False

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
        )

        return train_dataloader

    def val_dataloader(self):
        """Create dataloader for validation."""
        val_dataset = self.dataset["validation"]

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=(
                self.hparams.val_batch_size
                if self.hparams.val_batch_size
                else self.hparams.batch_size
            ),
            num_workers=self.hparams.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        return val_dataloader

    def test_dataloader(self):
        """Create dataloader for testing."""
        self.rouge_metrics = ["rouge1", "rouge2", "rougeL"]
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.rouge_metrics, use_stemmer=True
        )

        self.hparams.test_batch_size = (
            self.hparams.test_batch_size
            if self.hparams.test_batch_size
            else self.hparams.batch_size
        )

        test_dataset = self.dataset["test"]

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        return test_dataloader

    def configure_optimizers(self):
        """
        Configure the optimizers. Returns the optimizer and scheduler specified by
        the values in ``self.hparams``.
        """
        # create the train dataloader so the number of examples can be determined
        self.train_dataloader_object = self.train_dataloader()

        return generic_configure_optimizers(
            self.hparams, self.train_dataloader_object, self.named_parameters()
        )

    def calculate_loss(self, prediction_scores, labels):
        masked_lm_loss = self.loss_func(
            prediction_scores.view(-1, self.model.config.vocab_size), labels.view(-1)
        )
        return masked_lm_loss

    def _step(self, batch):
        """
        Perform a generic step of the model. Pass the batch through the model
        and return the loss.
        """
        source, target, source_mask, target_mask = (
            batch["source"],
            batch["target"],
            batch["source_mask"],
            batch["target_mask"],
        )

        labels = target.clone()
        # Padding token is ignored in loss function so below line is unnecessary.
        # labels[labels == 1] = -100  # -100 index = padding token

        outputs = self.forward(source, target, source_mask, target_mask, labels=labels)
        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):  # skipcq: PYL-W0613
        """Training step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.training_step>`__"""  # noqa: E501
        cross_entropy_loss = self._step(batch)

        self.log("train_loss", cross_entropy_loss, prog_bar=True)

        return cross_entropy_loss

    def validation_step(self, batch, batch_idx):  # skipcq: PYL-W0613
        """Validation step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_step>`__"""  # noqa: E501
        cross_entropy_loss = self._step(batch)
        self.log("val_loss", cross_entropy_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):  # skipcq: PYL-W0613
        """
        Test step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_step>`__
        Similar to :meth:`~abstractive.AbstractiveSummarizer.validation_step` in that in runs the inputs
        through the model. However, this method also calculates the ROUGE scores for each example-summary
        pair.
        """  # noqa: E501
        source_ids, target_ids, source_mask, _ = (
            batch["source"],
            batch["target"],
            batch["source_mask"],
            batch["target_mask"],
        )

        source_ids, source_mask = trim_batch(
            source_ids, self.tokenizer.pad_token_id, attention_mask=source_mask
        )
        target_ids = trim_batch(target_ids, self.tokenizer.pad_token_id)

        # Generate
        # Set `pad_token_id` to `self.target_eoseq_token_id`, which is the same as
        # `eos_token_id` in order to skip a warning. The `generate` function will
        # do this if we don't, but when we do it the warning does not occur.
        t0 = time()
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=5,
            # decoder_start_token_id=self.target_boseq_token_id,
            # bos_token_id=self.target_boseq_token_id,
            # eos_token_id=self.target_eoseq_token_id,
            # pad_token_id=self.target_eoseq_token_id,
            max_length=(
                self.hparams.gen_max_len
                if self.hparams.gen_max_len
                else int(self.tokenizer.model_max_length / 2)
            ),
            no_repeat_ngram_size=3,
            use_cache=True,
        )
        generation_time = time() - t0
        logger.debug("Generation Time: %.2f", generation_time)

        generated_ids = generated_ids.tolist()
        target_ids = target_ids.tolist()

        predictions = self.ids_to_clean_text(generated_ids, replace_sep_with_q=True)
        targets = self.ids_to_clean_text(target_ids, replace_sep_with_q=True)

        rouge_outputs = []
        if self.hparams.test_use_pyrouge:
            with open("save_gold.txt", "a+") as save_gold, open(
                "save_pred.txt", "a+"
            ) as save_pred:
                for i, _ in enumerate(targets):
                    save_gold.write(targets[i].strip() + "\n")
                for i, _ in enumerate(predictions):
                    save_pred.write(predictions[i].strip() + "\n")
        else:
            for target, prediction in zip(targets, predictions):
                target.replace("<q>", "\n")
                prediction.replace("<q>", "\n")
                rouge_outputs.append(self.rouge_scorer.score(target, prediction))

        # Save about `self.hparams.save_percentage` of the predictions and targets
        # if `self.hparams.save_percentage` is set.
        if (
            self.hparams.save_percentage
            and random.random() < self.hparams.save_percentage
        ):
            index_to_select = random.randrange(0, self.hparams.test_batch_size, 1)
            output_prediction = predictions[index_to_select]
            output_target = targets[index_to_select]
        else:
            output_prediction = None
            output_target = None

        output = OrderedDict(
            {
                "rouge_scores": rouge_outputs,
                "generation_time": generation_time,
                "prediction": output_prediction,
                "target": output_target,
            }
        )
        return output

    def test_epoch_end(self, outputs):
        """
        Called at the end of a testing epoch: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_epoch_end>`__
        Finds the mean of all the metrics logged by :meth:`~abstractive.AbstractiveSummarizer.test_step`.
        """  # noqa: E501
        avg_generation_time = np.array([x["generation_time"] for x in outputs]).mean()

        rouge_scores_log = {}

        if self.hparams.test_use_pyrouge:
            test_rouge("tmp", "save_pred.txt", "save_gold.txt")
        else:
            aggregator = scoring.BootstrapAggregator()
            rouge_scores_list = [
                rouge_score_set
                for batch_list in outputs
                for rouge_score_set in batch_list["rouge_scores"]
            ]
            for score in rouge_scores_list:
                aggregator.add_scores(score)
            # The aggregator returns a dictionary with keys coresponding to the rouge metric
            # and values that are `AggregateScore` objects. Each `AggregateScore` object is a
            # named tuple with a low, mid, and high value. Each value is a `Score` object, which
            # is also a named tuple, that contains the precision, recall, and fmeasure values.
            # For more info see the source code:
            # https://github.com/google-research/google-research/blob/master/rouge/scoring.py
            rouge_result = aggregator.aggregate()

            for metric, value in rouge_result.items():
                rouge_scores_log[metric + "-precision"] = value.mid.precision
                rouge_scores_log[metric + "-recall"] = value.mid.recall
                rouge_scores_log[metric + "-fmeasure"] = value.mid.fmeasure

        # Write the saved predictions and targets to file
        if self.hparams.save_percentage:
            predictions = [
                x["prediction"] for x in outputs if x["prediction"] is not None
            ]
            targets = [x["target"] for x in outputs if x["target"] is not None]

            if self.hparams.default_root_dir is None:
                save_dir = "."
            else:
                save_dir = self.hparams.default_root_dir

            output_test_predictions_file = os.path.join(
                save_dir, "test_predictions.txt"
            )
            output_test_targets_file = os.path.join(save_dir, "test_targets.txt")
            with open(output_test_predictions_file, "w+") as p_writer, open(
                output_test_targets_file, "w+"
            ) as t_writer:
                for prediction, target in zip(predictions, targets):
                    p_writer.writelines(s + "\n" for s in prediction)
                    t_writer.writelines(s + "\n" for s in target)
                p_writer.close()
                t_writer.close()

        # Generate logs
        tqdm_dict = {"generation_time": avg_generation_time}
        log = {**rouge_scores_log, **tqdm_dict}
        result = {"progress_bar": tqdm_dict, "log": log}
        return result

    def predict(self, input_sequence):
        """Summaries ``input_sequence`` using the model. Can summarize a list of
        sequences at once.

        Args:
            input_sequence (str or list[str]): The text to be summarized.

        Returns:
            str or list[str]: The summary text.
        """
        # If a single string is passed, wrap it in a list so `batch_encode_plus()`
        # processes it correctly
        if type(input_sequence) is str:
            input_sequence = [input_sequence]

        input_sequence_encoded = self.tokenizer.batch_encode_plus(
            input_sequence,
            pad_to_max_length=False,
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        input_sequence_encoded = torch.tensor(input_sequence_encoded)

        # If using the LongformerEncoderDecoder then apply the padding for sliding
        # chunks attention.
        if any(
            x in self.hparams.model_name_or_path.lower()
            for x in ["led-large", "led-base"]
        ):
            input_sequence_encoded = pad_tensors(
                input_sequence_encoded,
                nearest_multiple_of=self.model.config.attention_window[0],
            )

        t0 = time()
        generated_ids = self.model.generate(
            input_ids=input_sequence_encoded,
            num_beams=3,
            decoder_start_token_id=self.target_boseq_token_id,
            bos_token_id=self.target_boseq_token_id,
            eos_token_id=self.target_eoseq_token_id,
            pad_token_id=self.target_eoseq_token_id,
            max_length=(
                self.hparams.gen_max_len
                if self.hparams.gen_max_len
                else int(self.tokenizer.model_max_length / 2)
            ),
            no_repeat_ngram_size=3,
            use_cache=True,
        )
        generation_time = time() - t0
        logger.debug("Generation Time: %.2f", generation_time)

        generated_ids = generated_ids.tolist()
        prediction = self.ids_to_clean_text(generated_ids)

        return prediction

    def ids_to_clean_text(self, generated_ids, replace_sep_with_q=False):
        """Convert IDs generated from ``tokenizer.encode`` to a string using
        ``tokenizer.batch_decode`` and also clean up spacing and special tokens.

        Args:
            generated_ids (list): A list examples where each example is a list of
                IDs generated from ``tokenizer.encode``.
            replace_sep_with_q (bool, optional): Replace the ``self.tokenizer.sep_token``
                with "<q>". Useful for determineing sentence boundaries and calculating
                ROUGE scores. Defaults to False.

        Returns:
            list or string: A list of examples where each example is a string or just one
            string if only one example was passed to this function.
        """

        if replace_sep_with_q:
            generated_ids = (
                [
                    self.rouge_sentence_split_token_id
                    if token == self.tokenizer.sep_token_id
                    else token
                    for token in example_ids
                ]
                for example_ids in generated_ids
            )

        gen_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if len(gen_texts) == 1:
            return gen_texts[0]

        return list(map(str.strip, gen_texts))

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        """Save the model in the ``huggingface/transformers`` format when a checkpoint is saved."""
        if self.hparams.save_hg_transformer:
            save_path = os.path.join(self.hparams.weights_save_path, "best_tfmr")

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Arguments specific to this model"""
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="bert-base-uncased",
            help="Path to pre-trained model or shortcut name. A list of shortcut names can "
            + "be found at https://huggingface.co/transformers/pretrained_models.html. "
            + "Community-uploaded models are located at https://huggingface.co/models. "
            + "Default is 'bert-base-uncased'.",
        )
        parser.add_argument(
            "--decoder_model_name_or_path",
            type=str,
            default=None,
            help="Path to pre-trained model or shortcut name to use as the decoder if an "
            + "EncoderDecoderModel architecture is desired. If this option is not specified, "
            + "the shortcut name specified by `--model_name_or_path` is loaded using the "
            + "Seq2seq AutoModel. Default is 'bert-base-uncased'.",
        )
        parser.add_argument(
            "--batch_size",
            default=4,
            type=int,
            help="Batch size per GPU/CPU for training/evaluation/testing.",
        )
        parser.add_argument(
            "--val_batch_size",
            default=None,
            type=int,
            help="Batch size per GPU/CPU for evaluation. This option overwrites `--batch_size` "
            + "for evaluation only.",
        )
        parser.add_argument(
            "--test_batch_size",
            default=None,
            type=int,
            help="Batch size per GPU/CPU for testing. This option overwrites `--batch_size` for "
            + "testing only.",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            default=3,
            type=int,
            help="The number of workers to use when loading data. A general place to start is "
            + "to set num_workers equal to the number of CPUs on your machine. "
            + "More details here: https://pytorch-lightning.readthedocs.io/en/latest/performance.html#num-workers",  # noqa: E501
        )
        parser.add_argument(
            "--only_preprocess",
            action="store_true",
            help="Only preprocess and write the data to disk. Don't train model.",
        )
        parser.add_argument(
            "--no_prepare_data",
            action="store_true",
            help="Don't download, tokenize, or prepare data. Only load it from files.",
        )
        parser.add_argument(
            "--dataset",
            nargs="+",
            default="cnn_dailymail",
            help="The dataset name from the `nlp` library or a list of paths to Apache Arrow "
            + "files (that can be loaded with `nlp`) in the order train, validation, test to "
            + "use for training/evaluation/testing. Paths must contain a '/' to be interpreted "
            + "correctly. Default is `cnn_dailymail`.",
        )
        parser.add_argument(
            "--dataset_version",
            type=str,
            default="3.0.0",
            help="The version of the dataset specified by `--dataset`.",
        )
        parser.add_argument(
            "--data_example_column",
            type=str,
            default="article",
            help="The column of the `nlp` dataset that contains the text to be summarized. "
            + "Default value is for the `cnn_dailymail` dataset.",
        )
        parser.add_argument(
            "--data_summarized_column",
            type=str,
            default="highlights",
            help="The column of the `nlp` dataset that contains the summarized text. "
            + "Default value is for the `cnn_dailymail` dataset.",
        )
        parser.add_argument(
            "--cache_file_path",
            type=str,
            default=".",
            help="Path to cache the tokenized dataset.",
        )
        parser.add_argument(
            "--split_char",
            type=str,
            default=None,
            help="""If the `--data_summarized_column` is already split into sentences then use
            this option to specify which token marks sentence boundaries. If the summaries are
            not split into sentences then spacy will be used to split them. The default is None,
            which means to use spacy.""",
        )
        parser.add_argument(
            "--use_percentage_of_data",
            type=float,
            default=False,
            help="When filtering the dataset, only save a percentage of the data. This is "
            + "useful for debugging when you don't want to process the entire dataset.",
        )
        parser.add_argument(
            "--save_percentage",
            type=float,
            default=0.01,
            help="""Percentage (divided by batch_size) between 0 and 1 of the predicted and target
            summaries from the test set to save to disk during testing. This depends on batch
            size: one item from each batch is saved `--save_percentage` percent of the time.
            Thus, you can expect `len(dataset)*save_percentage/batch_size` summaries to be
            saved.""",
        )
        parser.add_argument(
            "--save_hg_transformer",
            action="store_true",
            help="Save the `huggingface/transformers` model whenever a checkpoint is saved.",
        )
        parser.add_argument(
            "--test_use_pyrouge",
            action="store_true",
            help="""Use `pyrouge`, which is an interface to the official ROUGE software, instead of
            the pure-python implementation provided by `rouge-score`. You must have the real ROUGE
            package installed. More details about ROUGE 1.5.5 here: https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5.
            It is recommended to use this option for official scores. The `ROUGE-L` measurements
            from `pyrouge` are equivalent to the `rougeLsum` measurements from the default
            `rouge-score` package.""",  # noqa: E501
        )
        parser.add_argument(
            "--sentencizer",
            action="store_true",
            help="Use a spacy sentencizer instead of a statistical model for sentence "
            + "detection (much faster but less accurate) during data preprocessing; see "
            + "https://spacy.io/api/sentencizer.",
        )
        parser.add_argument(
            "--model_max_length",
            type=int,
            default=None,
            help="Changes the `model_max_length` attribute of the tokenizer. Overrides the "
            + "default length of input sequences generated during data processing.",
        )
        parser.add_argument(
            "--gen_max_len",
            type=int,
            default=None,
            help="Maximum sequence length during generation while testing and when using the "
            + "`predict()` function.",
        )
        parser.add_argument(
            "--label_smoothing",
            type=float,
            default=0.1,
            help="`LabelSmoothingLoss` implementation from OpenNMT (https://bit.ly/2ObgVPP) as "
            + "stated in the original paper https://arxiv.org/abs/1512.00567.",
        )
        parser.add_argument(
            "--sortish_sampler",
            action="store_true",
            help="""Reorganize the input_ids by length with a bit of randomness. This can help
            to avoid memory errors caused by large batches by forcing large batches to be
            processed first.""",
        )
        parser.add_argument(
            "--nlp_cache_dir",
            type=str,
            default="~/nlp",
            help="Directory to cache datasets downloaded using `nlp`. Defaults to '~/nlp'.",
        )
        parser.add_argument(
            "--tie_encoder_decoder",
            action="store_true",
            help="Tie the encoder and decoder weights. Only takes effect when using an "
            + "EncoderDecoderModel architecture with the `--decoder_model_name_or_path` "
            + "option. Specifying this option is equivalent to the 'share' architecture "
            + "tested in 'Leveraging Pre-trained Checkpoints for Sequence Generation Tasks' "
            + "(https://arxiv.org/abs/1907.12461).",
        )

        return parser
