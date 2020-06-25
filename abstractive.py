import os
import logging
import random
import torch
import nlp
import itertools
import spacy
from spacy.lang.en import English
from functools import partial
from time import time
from collections import OrderedDict
from argparse import ArgumentParser
from torch import nn, optim
from rouge_score import rouge_scorer, scoring
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
import pytorch_lightning as pl
from transformers import (
    BertForMaskedLM,
    BertModel,
    AutoTokenizer,
    EncoderDecoderModel,
    BartTokenizer,
)
from helpers import lr_lambda_func, pad
from convert_to_extractive import tokenize

logger = logging.getLogger(__name__)

try:
    from longbart import LongBartForConditionalGeneration
except ImportError:
    logger.warn(
        "Abstractive Only: Could not import `LongBartForConditionalGeneration` from `longbart`, which means the `longbart` model is not available. Install with `pip install git+https://github.com/patil-suraj/longbart.git`."
    )


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by ``pad_token_id``."""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class AbstractiveSummarizer(pl.LightningModule):
    """
    A machine learning model that abstractively summarizes an input text using a seq2seq model.
    Main class that handles the data loading, initial processing, training/testing/validating setup,
    and contains the actual model.
    """

    def __init__(self, hparams):
        super(AbstractiveSummarizer, self).__init__()

        self.hparams = hparams

        if "longbart" in self.hparams.model_name_or_path.lower():
            self.model = LongBartForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path
            )
            self.tokenizer = BartTokenizer.from_pretrained(
                self.hparams.model_name_or_path
            )
        else:
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.hparams.model_name_or_path,
                (
                    self.hparams.decoder_model_name_or_path
                    if self.hparams.decoder_model_name_or_path
                    else self.hparams.model_name_or_path
                ),
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.model_name_or_path, use_fast=True
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

        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(
        self, source=None, target=None, source_mask=None, target_mask=None, labels=None
    ):
        """Model forward function. See the `60 minute bliz tutorial <https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html>`_
        if you are unsure what a forward function is.

        Args:
            source (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, optional): Indices 
                of input sequence tokens in the vocabulary for the encoder. 
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
        """
        # `self.model.forward()` returns `decoder_outputs + encoder_outputs`
        outputs = self.model.forward(
            input_ids=source,
            attention_mask=source_mask,
            decoder_input_ids=target,
            decoder_attention_mask=target_mask,
            labels=labels,
        )

        cross_entropy_loss, prediction_scores = outputs[:2]
        return cross_entropy_loss, prediction_scores

    def prepare_data(self):
        """
        Create the data using the ``huggingface/nlp`` library. This function handles
        downloading, preprocessing, tokenization, and feature extraction.
        """

        # load spacy english small model with the "tagger" and "ner" disabled since
        # we only need the "tokenizer" and "parser"
        # more info: https://spacy.io/usage/processing-pipelines
        if self.hparams.sentencizer:
            spacy_nlp = English()
            sentencizer = spacy_nlp.create_pipe("sentencizer")
            spacy_nlp.add_pipe(sentencizer)
        else:
            spacy_nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])

        def convert_to_features(example_batch):
            max_length = self.tokenizer.max_len

            articles = example_batch[self.hparams.data_example_column]
            articles_encoded = self.tokenizer.batch_encode_plus(
                articles, pad_to_max_length=True, truncation=True,
            )

            highlights = example_batch[self.hparams.data_summarized_column]
            # Tokenize highlights using spacy to split them into sentences
            highlights_tokenized = tokenize(
                spacy_nlp, highlights, disable_progress_bar=True
            )
            sep_token = self.tokenizer.sep_token

            highlights_input_ids = []
            highlights_attention_masks = []
            # For each ground-truth summary
            for highlight in highlights_tokenized:
                # `highlight` is a list of sentences where each sentence is a list of tokens
                # Combine those tokens to create a list of sentences.
                sents = [" ".join(list_of_ids) for list_of_ids in highlight]
                # Tokenize each sentence and append the `sep_token`
                sents_tokenized = []
                for sent in sents:
                    sent = self.tokenizer.tokenize(sent)
                    sent.append(sep_token)
                    sents_tokenized.append(sent)

                # Delete the last `sep_token` from the last sentence
                del sents_tokenized[-1][-1]
                # Flatten `sents_tokenized` (a list of sentences where each sentence is a list
                # of tokens) to a list of tokens
                sents_tokenized_flat = list(
                    itertools.chain.from_iterable(sents_tokenized)
                )

                # Convert the tokens to `input_ids`
                # `max_length` is the max length minus 2 because we need to add the
                # beginning and ending tokens to the target
                sents_input_ids = self.tokenizer.encode_plus(
                    sents_tokenized_flat,
                    truncation=True,
                    is_pretokenized=True,
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
                highlights_input_ids, self.tokenizer.pad_token_id, width=max_length,
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

        self.dataset = nlp.load_dataset(
            self.hparams.dataset, self.hparams.dataset_version
        )

        self.dataset["train"] = self.dataset["train"].map(
            convert_to_features,
            batched=True,
            cache_file_name=os.path.join(
                self.hparams.cache_file_path, "train_tokenized"
            ),
        )
        self.dataset["validation"] = self.dataset["validation"].map(
            convert_to_features,
            batched=True,
            cache_file_name=os.path.join(
                self.hparams.cache_file_path, "validation_tokenized"
            ),
        )
        self.dataset["test"] = self.dataset["test"].map(
            convert_to_features,
            batched=True,
            cache_file_name=os.path.join(
                self.hparams.cache_file_path, "test_tokenized"
            ),
        )

        columns = ["source", "target", "source_mask", "target_mask"]
        self.dataset["train"].set_format(type="torch", columns=columns)
        self.dataset["validation"].set_format(type="torch", columns=columns)
        self.dataset["test"].set_format(type="torch", columns=columns)

    def train_dataloader(self):
        """Create dataloader for training."""
        train_dataset = self.dataset["train"]

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            pin_memory=True,
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
        )

        return test_dataloader

    def configure_optimizers(self):
        """
        Configure the optimizers. Returns the optimizer and scheduler specified by
        the values in ``self.hparams``.
        """
        # create the train dataloader so the number of examples can be determined
        self.train_dataloader_object = self.train_dataloader()
        # check that max_steps is not None and is greater than 0
        if self.hparams.max_steps and self.hparams.max_steps > 0:
            # pytorch_lightning steps the scheduler every batch but only updates
            # the global_step every gradient accumulation cycle. Therefore, the
            # scheduler needs to have `accumulate_grad_batches` * `max_steps` in
            # order to reach `max_steps`.
            # See: https://github.com/PyTorchLightning/pytorch-lightning/blob/f293c9b5f4b4f9fabb2eec0c369f08a66c57ef14/pytorch_lightning/trainer/training_loop.py#L624
            t_total = self.hparams.max_steps * self.hparams.accumulate_grad_batches
        else:
            t_total = int(
                len(self.train_dataloader_object)
                * self.hparams.max_epochs
                // self.hparams.accumulate_grad_batches
            )
            if self.hparams.overfit_pct > 0.0:
                t_total = int(t_total * self.hparams.overfit_pct)

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

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        if self.hparams.use_scheduler:
            if self.hparams.use_scheduler == "linear":
                # We have to import the function and create a partial because functions cannot be
                # serialized by python pickle. Therefore, if the normal `get_linear_schedule_with_warmup`
                # function provided by `transformers` was used, the program would fail to save
                # `self.hparams` because the optimizer would contain a locale function that cannot be
                # pickled.
                lr_lambda = partial(
                    lr_lambda_func,
                    num_warmup_steps=self.hparams.warmup_steps
                    * self.hparams.accumulate_grad_batches,
                    num_training_steps=t_total,
                )
                # multiply by `hparams.accumulate_grad_batches` above because pytorch_lightning
                # steps are for each batch, except for the `trainer.global_step`, which tracks
                # the actual number of steps

                scheduler = LambdaLR(optimizer, lr_lambda, -1)

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
            # the below interval is called "step" but the scheduler is moved forward
            # every batch.
            scheduler_dict = {"scheduler": scheduler, "interval": "step"}

            return ([optimizer], [scheduler_dict])
        else:
            return optimizer

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
        target, target_mask = trim_batch(
            target, self.tokenizer.pad_token_id, target_mask
        )

        labels = target.clone()
        labels[labels == 0] = -100  # -100 index = padding token
        outputs = self.forward(source, target, source_mask, target_mask, labels=labels)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """Training step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.training_step>`__"""
        cross_entropy_loss = self._step(batch)

        tqdm_dict = {"train_loss": cross_entropy_loss}
        output = OrderedDict(
            {"loss": cross_entropy_loss, "progress_bar": tqdm_dict, "log": tqdm_dict,}
        )
        return output

    def validation_step(self, batch, batch_idx):
        """Validation step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_step>`__"""
        cross_entropy_loss = self._step(batch)

        tqdm_dict = {"val_loss": cross_entropy_loss}
        output = OrderedDict(
            {
                "val_loss": cross_entropy_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of a validation epoch: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_epoch_end>`__
        Finds the mean of all the metrics logged by :meth:`~abstractive.AbstractiveSummarizer.validation_step`.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tqdm_dict = {"val_loss": avg_loss}
        output = {
            "val_loss": avg_loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }
        return output

    def test_step(self, batch, batch_idx):
        """
        Test step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_step>`__
        Similar to :meth:`~abstractive.AbstractiveSummarizer.validation_step` in that in runs the inputs
        through the model. However, this method also calculates the ROUGE scores for each example-summary
        pair.
        """
        source_ids, target_ids, source_mask, _ = batch.values()

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
            decoder_start_token_id=self.target_boseq_token_id,
            bos_token_id=self.target_boseq_token_id,
            eos_token_id=self.target_eoseq_token_id,
            pad_token_id=self.target_eoseq_token_id,
            max_length=(
                self.hparams.gen_max_len
                if self.hparams.gen_max_len
                else self.tokenizer.max_len / 2
            ),
            no_repeat_ngram_size=3,
            use_cache=True,
        )
        generation_time = time() - t0
        logger.debug("Generation Time: {}".format(generation_time))

        generated_ids = generated_ids.tolist()
        target_ids = target_ids.tolist()

        predictions = self.ids_to_clean_text(generated_ids, replace_sep_with_q=True)
        targets = self.ids_to_clean_text(target_ids, replace_sep_with_q=True)

        cross_entropy_loss, prediction_scores = self.forward(**batch)

        rouge_outputs = []
        if self.hparams.test_use_pyrouge:
            with open("save_gold.txt", "a+") as save_gold, open(
                "save_pred.txt", "a+"
            ) as save_pred:
                for i in range(len(targets)):
                    save_gold.write(targets[i].strip() + "\n")
                for i in range(len(predictions)):
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
        """
        avg_generation_time = torch.stack(
            [x["generation_time"] for x in outputs]
        ).mean()

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
            # For more info see the source code: https://github.com/google-research/google-research/blob/master/rouge/scoring.py
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
            output_test_predictions_file = os.path.join(
                self.hparams.default_root_dir, "test_predictions.txt"
            )
            output_test_targets_file = os.path.join(
                self.hparams.default_root_dir, "test_targets.txt"
            )
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
                else self.tokenizer.max_len / 2
            ),
            no_repeat_ngram_size=3,
            use_cache=True,
        )
        generation_time = time() - t0
        logger.debug("Generation Time: {}".format(generation_time))

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
            gen_texts = []
            for ids in generated_ids:
                # Removal of special tokens except `self.tokenizer.sep_token_id`
                tokens = []
                for index in ids:
                    index = int(index)
                    # If the current token is `tokenizer.sep_token` then set it to "<q>"
                    if index == self.tokenizer.sep_token_id:
                        tokens.append("<q>")
                    elif index in self.tokenizer.all_special_ids:
                        continue
                    else:
                        current_token = self.tokenizer._convert_id_to_token(index)
                        tokens.append(current_token)

                gen_text_messy = self.tokenizer.convert_tokens_to_string(tokens)

                gen_text = self.tokenizer.clean_up_tokenization(gen_text_messy)

                gen_texts.append(gen_text)
                print(gen_text)

            return gen_texts

        else:
            gen_texts = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            if len(gen_texts) == 1:
                return gen_texts[0]
            else:
                return list(map(str.strip, gen_text))

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
            help="Path to pre-trained model or shortcut name. A list of shortcut names can be found at https://huggingface.co/transformers/pretrained_models.html. Community-uploaded models are located at https://huggingface.co/models.",
        )
        parser.add_argument(
            "--decoder_model_name_or_path",
            type=str,
            default=None,
            help="Path to pre-trained model or shortcut name to use as the decoder. Default is the value of `--model_name_or_path`.",
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
            help="Batch size per GPU/CPU for evaluation. This option overwrites `--batch_size` for evaluation only.",
        )
        parser.add_argument(
            "--test_batch_size",
            default=None,
            type=int,
            help="Batch size per GPU/CPU for testing. This option overwrites `--batch_size` for testing only.",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            default=3,
            type=int,
            help="The number of workers to use when loading data. A general place to start is to set num_workers equal to the number of CPUs on your machine. More details here: https://pytorch-lightning.readthedocs.io/en/latest/performance.html#num-workers",
        )
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
        parser.add_argument("--weight_decay", default=1e-2, type=float)
        parser.add_argument(
            "--dataset",
            type=str,
            default="cnn_dailymail",
            help="The dataset name from the `nlp` library to use for training/evaluation/testing. Default is `cnn_dailymail`.",
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
            help="The column of the `nlp` dataset that contains the text to be summarized. Default value is for the `cnn_dailymail` dataset.",
        )
        parser.add_argument(
            "--data_summarized_column",
            type=str,
            default="highlights",
            help="The column of the `nlp` dataset that contains the summarized text. Default value is for the `cnn_dailymail` dataset.",
        )
        parser.add_argument(
            "--cache_file_path",
            type=str,
            default=".",
            help="Path to cache the tokenized dataset.",
        )
        parser.add_argument(
            "--save_percentage",
            type=float,
            default=0.01,
            help="""Percentage (divided by batch_size) between 0 and 1 of the predicted and target 
            summaries from the test set to save to disk during testing. This depends on batch 
            size: one item from each batch is saved `--save_percentage` percent of the time. 
            Thus, you can expect `len(dataset)*save_percentage/batch_size` summaries to be saved.""",
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
            `rouge-score` package.""",
        )
        parser.add_argument(
            "--sentencizer",
            action="store_true",
            help="Use a spacy sentencizer instead of a statistical model for sentence detection (much faster but less accurate) during data preprocessing; see https://spacy.io/api/sentencizer.",
        )
        parser.add_argument(
            "--gen_max_len",
            type=int,
            default=None,
            help="Maximum sequence length during generation while testing and when using the `predict()` function.",
        )

        return parser
