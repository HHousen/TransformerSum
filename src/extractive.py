import glob
import logging
import os
import statistics
import sys
import types
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from functools import partial
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from rouge_score import rouge_scorer, scoring
from spacy.lang.en import English
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.data.metrics import acc_and_f1

from classifier import (
    LinearClassifier,
    SimpleLinearClassifier,
    TransformerEncoderClassifier,
)
from data import FSDataset, FSIterableDataset, SentencesProcessor, pad_batch_collate
from helpers import block_trigrams, generic_configure_optimizers, load_json, test_rouge
from pooling import Pooling

logger = logging.getLogger(__name__)


# CUSTOM_MODEL_CLASSES = ("longformer",)

try:
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    MODEL_CLASSES = tuple(MODEL_MAPPING_NAMES.keys())  # + CUSTOM_MODEL_CLASSES
except ImportError:
    logger.warning(
        "Could not import `MODEL_MAPPING_NAMES` from transformers because it is an old version."
    )

    MODEL_CLASSES = (
        tuple(
            "Note: Only showing custom models because old version of `transformers` detected."
        )
        # + CUSTOM_MODEL_CLASSES
    )


def longformer_modifier(final_dictionary):
    """
    Creates the ``global_attention_mask`` for the longformer. Tokens with global attention
    attend to all other tokens, and all other tokens attend to them. This is important for
    task-specific finetuning because it makes the model more flexible at representing the
    task. For example, for classification, the `<s>` token should be given global attention.
    For QA, all question tokens should also have global attention. For summarization,
    global attention is given to all of the `<s>` (RoBERTa 'CLS' equivalent) tokens. Please
    refer to the `Longformer paper <https://arxiv.org/abs/2004.05150>`_ for more details. Mask
    values selected in ``[0, 1]``: ``0`` for local attention, ``1`` for global attention.
    """
    # `batch_size` is the number of attention masks (one mask per input sequence)
    batch_size = len(final_dictionary["attention_mask"])
    # `sequence_length` is the number of tokens for the first sequence in the batch
    sequence_length = len(final_dictionary["attention_mask"][0])
    # create `global_attention_mask` using the above details
    global_attention_mask = torch.tensor([[0] * sequence_length] * batch_size)
    # set the `sent_rep_token_ids` to 1, which is global attention
    for idx, items in enumerate(final_dictionary["sent_rep_token_ids"]):
        global_attention_mask[idx, items] = 1

    final_dictionary["global_attention_mask"] = global_attention_mask
    # The `global_attention_mask` is passed through the model's `forward`
    # function as `**kwargs`.
    return final_dictionary


class ExtractiveSummarizer(pl.LightningModule):
    """
    A machine learning model that extractively summarizes an input text by scoring the sentences.
    Main class that handles the data loading, initial processing, training/testing/validating setup,
    and contains the actual model.
    """

    def __init__(self, hparams, embedding_model_config=None, classifier_obj=None):
        super(ExtractiveSummarizer, self).__init__()

        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)

        # Set new parameters to defaults if they do not exist in the `hparams` Namespace
        hparams.gradient_checkpointing = getattr(
            hparams, "gradient_checkpointing", False
        )
        hparams.tokenizer_no_use_fast = getattr(hparams, "tokenizer_no_use_fast", False)
        hparams.data_type = getattr(hparams, "data_type", "none")

        self.save_hyperparameters(hparams)
        self.forward_modify_inputs_callback = None

        if not embedding_model_config:
            embedding_model_config = AutoConfig.from_pretrained(
                hparams.model_name_or_path,
                gradient_checkpointing=hparams.gradient_checkpointing,
            )

        self.word_embedding_model = AutoModel.from_config(embedding_model_config)

        if (
            any(
                x in hparams.model_name_or_path
                for x in ["roberta", "distil", "longformer"]
            )
        ) and not hparams.no_use_token_type_ids:
            logger.warning(
                (
                    "You are using a %s model but did not set "
                    + "--no_use_token_type_ids. This model does not support `token_type_ids` so "
                    + "this option has been automatically enabled."
                ),
                hparams.model_type,
            )
            self.hparams.no_use_token_type_ids = True

        self.emd_model_frozen = False
        if hparams.num_frozen_steps > 0:
            self.emd_model_frozen = True
            self.freeze_web_model()

        if hparams.pooling_mode == "sent_rep_tokens":
            self.pooling_model = Pooling(
                sent_rep_tokens=True, mean_tokens=False, max_tokens=False
            )
        elif hparams.pooling_mode == "max_tokens":
            self.pooling_model = Pooling(
                sent_rep_tokens=False, mean_tokens=False, max_tokens=True
            )
        else:
            self.pooling_model = Pooling(
                sent_rep_tokens=False, mean_tokens=True, max_tokens=False
            )

        # if a classifier object was passed when creating this model then store that as the
        # `encoder`
        if classifier_obj:
            self.encoder = classifier_obj
        # otherwise create the classifier using the `hparams.classifier` parameter if available
        # if the `hparams.classifier` parameter is missing then create a `LinearClassifier`
        else:
            # returns `classifier` value if it exists, otherwise returns False
            classifier_exists = getattr(hparams, "classifier", False)
            if (not classifier_exists) or (hparams.classifier == "linear"):
                self.encoder = LinearClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                )
            elif hparams.classifier == "simple_linear":
                self.encoder = SimpleLinearClassifier(
                    self.word_embedding_model.config.hidden_size
                )
            elif hparams.classifier == "transformer":
                self.encoder = TransformerEncoderClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                    num_layers=hparams.classifier_transformer_num_layers,
                )
            elif hparams.classifier == "transformer_linear":
                linear = LinearClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                )
                self.encoder = TransformerEncoderClassifier(
                    self.word_embedding_model.config.hidden_size,
                    dropout=hparams.classifier_dropout,
                    num_layers=hparams.classifier_transformer_num_layers,
                    custom_reduction=linear,
                )
            else:
                logger.error(
                    "%s is not a valid value for `--classifier`. Exiting...",
                    hparams.classifier,
                )
                sys.exit(1)

        # Set `hparams.no_test_block_trigrams` to False if it does not exist,
        # otherwise set its value to itself, resulting in no change
        self.hparams.no_test_block_trigrams = getattr(
            hparams, "no_test_block_trigrams", False
        )

        # BCELoss: https://pytorch.org/docs/stable/nn.html#bceloss
        # `reduction` is "none" so the mean can be computed with padding ignored.
        # `nn.BCEWithLogitsLoss` (which combines a sigmoid layer and the BCELoss
        # in one single class) is used because it takes advantage of the log-sum-exp
        # trick for numerical stability. Padding values are 0 and if 0 is the input
        # to the sigmoid function the output will be 0.5. This will cause issues when
        # inputs with more padding will have higher loss values. To solve this, all
        # padding values are set to -9e3 as the last step of each encoder. The sigmoid
        # function transforms -9e3 to nearly 0, thus preserving the proper loss
        # calculation. See `compute_loss()` for more info.
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

        # Data
        self.processor = SentencesProcessor(name="main_processor")

        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name
            if hparams.tokenizer_name
            else hparams.model_name_or_path,
            use_fast=(not self.hparams.tokenizer_no_use_fast),
        )

        self.train_dataloader_object = None  # not created yet
        self.datasets = None
        self.pad_batch_collate = None
        self.global_step_tracker = None
        self.rouge_metrics = None
        self.rouge_scorer = None

    def forward(
        self,
        input_ids,
        attention_mask,
        sent_rep_mask=None,
        token_type_ids=None,
        sent_rep_token_ids=None,
        sent_lengths=None,
        sent_lengths_mask=None,
        **kwargs,
    ):
        r"""Model forward function. See the `60 minute bliz tutorial <https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html>`_
        if you are unsure what a forward function is.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
                `What are input IDs? <https://huggingface.co/transformers/glossary.html#input-ids>`_
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token
                indices. Mask values selected in ``[0, 1]``: ``1`` for tokens that are NOT
                MASKED, ``0`` for MASKED tokens. `What are attention masks? <https://huggingface.co/transformers/glossary.html#attention-mask>`_
            sent_rep_mask (torch.Tensor, optional): Indicates which numbers in ``sent_rep_token_ids``
                are actually the locations of sentence representation ids and which are padding.
                Defaults to None.
            token_type_ids (torch.Tensor, optional): Usually, segment token indices to indicate
                first and second portions of the inputs. However, for summarization they are used
                to indicate different sentences. Depending on the size of the token type id vocabulary,
                these values may alternate between ``0`` and ``1`` or they may increase sequentially
                for each sentence in the input.. Defaults to None.
            sent_rep_token_ids (torch.Tensor, optional): The locations of the sentence representation
                tokens. Defaults to None.
            sent_lengths (torch.Tensor, optional):  A list of the lengths of each sentence in
                ``input_ids``. See :meth:`data.pad_batch_collate` for more info about the
                generation of thisfeature. Defaults to None.
            sent_lengths_mask (torch.Tensor, optional): Created on-the-fly by :meth:`data.pad_batch_collate`.
                Similar to ``sent_rep_mask``: ``1`` for value and ``0`` for padding. See
                :meth:`data.pad_batch_collate` for more info about the generation of this
                feature. Defaults to None.

        Returns:
            tuple: Contains the sentence scores and mask as ``torch.Tensor``\ s. The mask is either
            the ``sent_rep_mask`` or ``sent_lengths_mask`` depending on the pooling mode used
            during model initialization.
        """  # noqa: E501
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if not self.hparams.no_use_token_type_ids:
            inputs["token_type_ids"] = token_type_ids

        if self.forward_modify_inputs_callback:
            inputs = self.forward_modify_inputs_callback(inputs)  # skipcq: PYL-E1102

        outputs = self.word_embedding_model(**inputs, **kwargs)
        word_vectors = outputs[0]

        sents_vec, mask = self.pooling_model(
            word_vectors=word_vectors,
            sent_rep_token_ids=sent_rep_token_ids,
            sent_rep_mask=sent_rep_mask,
            sent_lengths=sent_lengths,
            sent_lengths_mask=sent_lengths_mask,
        )

        sent_scores = self.encoder(sents_vec, mask)
        return sent_scores, mask

    def unfreeze_web_model(self):
        """Un-freezes the ``word_embedding_model``"""
        for param in self.word_embedding_model.parameters():
            param.requires_grad = True

    def freeze_web_model(self):
        """Freezes the encoder ``word_embedding_model``"""
        for param in self.word_embedding_model.parameters():
            param.requires_grad = False

    def compute_loss(self, outputs, labels, mask):
        """Compute the loss between model outputs and ground-truth labels.

        Args:
            outputs (torch.Tensor): Output sentence scores obtained from
                :meth:`~extractive.ExtractiveSummarizer.forward`
            labels (torch.Tensor): Ground-truth labels (``1`` for sentences that should be in
                the summary, ``0`` otherwise) from the batch generated during the data
                preprocessing stage.
            mask (torch.Tensor): Mask returned by :meth:`~extractive.ExtractiveSummarizer.forward`,
                either ``sent_rep_mask`` or ``sent_lengths_mask`` depending on the pooling mode
                used during model initialization.

        Returns:
            [tuple]: Losses: (total_loss, total_norm_batch_loss, sum_avg_seq_loss,
                mean_avg_seq_loss, average_loss)
        """
        try:
            loss = self.loss_func(outputs, labels.float())
        except ValueError as e:
            logger.error(e)
            logger.error(
                "Details about above error:\n1. outputs=%s\n2. labels.float()=%s",
                outputs,
                labels.float(),
            )
            sys.exit(1)

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

    def setup(self, stage):
        """Download the `word_embedding_model` if the model will be trained."""
        # The model is having training resumed if the `hparams` contains `resume_from_checkpoint`
        # and `resume_from_checkpoint` is True.
        resuming = (
            hasattr(self.hparams, "resume_from_checkpoint")
            and self.hparams.resume_from_checkpoint
        )
        # `stage` can be "fit" or "test". Only load the pre-trained weights when
        # beginning to fit for the first time (when we are not resuming)
        if stage == "fit" and not resuming:
            logger.info("Loading `word_embedding_model` pre-trained weights.")
            self.word_embedding_model = AutoModel.from_pretrained(
                self.hparams.model_name_or_path, config=self.word_embedding_model.config
            )

    def json_to_dataset(
        self,
        tokenizer,
        hparams,
        inputs=None,
        num_files=0,
        processor=None,
    ):
        """Convert json output from ``convert_to_extractive.py`` to a ".pt" file containing
        lists or tensors using a :class:`data.SentencesProcessor`. This function is run by
        :meth:`~extractive.ExtractiveSummarizer.prepare_data` in parallel.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to convert examples
                to input_ids. Usually is ``self.tokenizer``.
            hparams (argparse.Namespace): Hyper-parameters used to create the model. Usually
                is ``self.hparams``.
            inputs (tuple, optional): (idx, json_file) Current loop index and path to json
                file. Defaults to None.
            num_files (int, optional): The total number of files to process. Used to display
                a nice progress indicator. Defaults to 0.
            processor (data.SentencesProcessor, optional): The :class:`data.SentencesProcessor`
                object to convert the json file to usable features. Defaults to None.
        """
        idx, json_file = inputs
        logger.info("Processing %s (%i/%i)", json_file, idx + 1, num_files)

        # open current json file (which is a set of documents)
        documents, file_path = load_json(json_file)

        all_sources = []
        all_ids = []
        all_targets = []
        for doc in documents:  # for each document in the json file
            source = doc["src"]
            if "tgt" in doc:
                target = doc["tgt"]
                all_targets.append(target)

            ids = doc["labels"]

            all_sources.append(source)
            all_ids.append(ids)

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
            create_source=all_targets,  # create the source if targets were present
            n_process=hparams.processing_num_threads,
            max_length=(
                hparams.max_seq_length
                if hparams.max_seq_length
                else self.tokenizer.model_max_length
            ),
            pad_on_left=self.tokenizer.padding_side == "left",
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            return_type="lists",
            save_to_path=hparams.data_path,
            save_to_name=os.path.basename(file_path),
            save_as_type=hparams.data_type,
        )

    def prepare_data(self):
        """
        Runs :meth:`~extractive.ExtractiveSummarizer.json_to_dataset` in parallel.
        :meth:`~extractive.ExtractiveSummarizer.json_to_dataset` is the function that actually
        loads and processes the examples as described below.
        Algorithm: For each json file outputted by the ``convert_to_extractive.py`` script:

        1. Load json file.
        2. Add each document in json file to ``SentencesProcessor`` defined in ``self.processor``,
            overwriting any previous data in the processor.
        3. Run :meth:`data.SentencesProcessor.get_features` to save the extracted features to disk
            as a ``.pt`` file containing a pickled python list of dictionaries, which each
            dictionary contains the extracted features.

        Memory Usage Note: If sharding was turned off during the ``convert_to_extractive`` process
        then this function will run once, loading the entire dataset into memory to process
        just like the ``convert_to_extractive.py`` script.
        """

        def get_inferred_data_type(dataset_files):
            dataset_files_extensions = [os.path.splitext(x)[1] for x in dataset_files]
            dataset_files_extensions_equal = len(set(dataset_files_extensions)) <= 1

            if (
                not dataset_files_extensions_equal
            ) and self.hparams.data_type == "none":
                logger.error(
                    "Cannot infer data file type because files with different extensions "
                    + "detected. Please set `--data_type`."
                )
                sys.exit(1)

            most_common = None
            if len(dataset_files_extensions) > 0:
                # If the most common file extension found is not the specified data type
                # then warn the user they may have chosen the wrong data type.
                most_common = statistics.mode(dataset_files_extensions)[1:]
                if (
                    most_common != self.hparams.data_type
                    and self.hparams.data_type != "none"
                ):
                    logger.warning(
                        "`--data_type` is '%s', but the most common file type detected in the "
                        + "`--data_path` is '%s'. Using '%s' as the type. Data will be processed "
                        + "if this type does not exist. Did you choose the correct data type?",
                        self.hparams.data_type,
                        most_common,
                        self.hparams.data_type,
                    )

            if len(dataset_files) == 0 and self.hparams.data_type == "none":
                logger.error(
                    "Data is going to be processed, but you have not specified an output format. "
                    + "Set `--data_type` to the desired format."
                )
                sys.exit(1)

            if self.hparams.data_type == "none":
                inferred_data_type = most_common
            else:
                inferred_data_type = self.hparams.data_type

            return inferred_data_type

        datasets = {}

        # loop through all data_splits
        data_splits = [
            self.hparams.train_name,
            self.hparams.val_name,
            self.hparams.test_name,
        ]
        for corpus_type in data_splits:
            # get the current list of dataset files. if preprocessing has already happened
            # then this will be the list of files that should be passed to an FSDataset.
            # if preprocessing has not happened then `dataset_files` should be an empty list
            # and the data will be processed
            dataset_files = glob.glob(
                os.path.join(self.hparams.data_path, "*" + corpus_type + ".*.*")
            )
            # remove json files from glob results since they are unprocessed files
            dataset_files = [x for x in dataset_files if "json" not in x]

            inferred_data_type = get_inferred_data_type(dataset_files)

            # rescan for dataset files after data type is determined
            dataset_files = glob.glob(
                os.path.join(
                    self.hparams.data_path,
                    "*" + corpus_type + ".*." + inferred_data_type,
                )
            )

            # if no dataset files detected or model is set to `only_preprocess`
            if (not dataset_files) or (self.hparams.only_preprocess):
                json_files = glob.glob(
                    os.path.join(self.hparams.data_path, "*" + corpus_type + ".*.json*")
                )
                if len(json_files) == 0:
                    logger.error(
                        "No JSON dataset files detected for %s split. Make sure the `--data_path`"
                        + " is correct.",
                        corpus_type,
                    )
                    sys.exit(1)

                if self.hparams.preprocess_resume:
                    completed_files = [
                        os.path.splitext(os.path.basename(i))[0] for i in dataset_files
                    ]
                    logger.info("Not Processing Shards: %s", completed_files)

                    def remove_complete(doc):
                        # if compression was enabled (files end in ".gz") then remove the ".gz"
                        if doc.endswith(".gz"):
                            doc = os.path.splitext(doc)[0]
                        # remove the ".json" extension
                        doc = os.path.splitext(os.path.basename(doc))[0]

                        # remove the document if it was already processed
                        if doc in completed_files:
                            return False  # remove
                        return True  # keep

                    json_files = list(filter(remove_complete, json_files))

                num_files = len(json_files)

                # pool = Pool(self.hparams.num_threads)
                json_to_dataset_processor = partial(
                    self.json_to_dataset,
                    self.tokenizer,
                    self.hparams,
                    num_files=num_files,
                    processor=self.processor,
                )

                for _ in map(
                    json_to_dataset_processor,
                    zip(range(len(json_files)), json_files),
                ):
                    pass
                # pool.close()
                # pool.join()

                # since the dataset has been prepared, the processed dataset files should
                # exist on disk. scan for final dataset files again.
                dataset_files = glob.glob(
                    os.path.join(
                        self.hparams.data_path,
                        "*" + corpus_type + ".*." + inferred_data_type,
                    )
                )

            # if set to only preprocess the data then continue to next loop
            # (aka next split of dataset)
            if self.hparams.only_preprocess:
                continue

            # always create actual dataset, either after writing the shard  files to disk
            # or by skipping that step (because preprocessed files detected) and going right to
            # loading.
            if self.hparams.dataloader_type == "map":
                if inferred_data_type != "txt":
                    logger.error(
                        """The `--dataloader_type` is 'map' but the `--data_type` was not
                        inferred to be 'txt'. The map-style dataloader requires 'txt' data.
                        Either set `--dataloader_type` to 'iterable' to use the old data
                        format or process the JSON to TXT by setting `--data_type` to
                        'txt'. Alternatively, you can convert directly from PT to TXT
                        using `scripts/convert_extractive_pt_to_txt.py`."""
                    )
                    sys.exit(1)
                datasets[corpus_type] = FSDataset(dataset_files, verbose=True)
            elif self.hparams.dataloader_type == "iterable":
                # Since `FSIterableDataset` is an `IterableDataset` the `DataLoader` will ask
                # the `Dataset` for the length instead of calculating it because the length
                # of `IterableDatasets` might not be known, but it is in this case.
                datasets[corpus_type] = FSIterableDataset(dataset_files, verbose=True)
                # Force use one worker if using an iterable dataset to prevent duplicate data
                self.hparams.dataloader_num_workers = 1

        # if set to only preprocess the data then exit after all loops have been completed
        if self.hparams.only_preprocess:
            logger.warning(
                "Exiting since data has been preprocessed and written to disk "
                + "and `hparams.only_preprocess` is True."
            )
            sys.exit(0)

        self.datasets = datasets

        # Create `pad_batch_collate` function
        # If the model is a longformer then create the `global_attention_mask`
        if self.hparams.model_type == "longformer":

            self.pad_batch_collate = partial(
                pad_batch_collate, modifier=longformer_modifier
            )
        else:
            # default is to just use the normal `pad_batch_collate` function
            self.pad_batch_collate = pad_batch_collate

    def train_dataloader(self):
        """Create dataloader for training if it has not already been created."""
        if self.train_dataloader_object:
            return self.train_dataloader_object
        if not hasattr(self, "datasets"):
            self.prepare_data()
        self.global_step_tracker = 0

        train_dataset = self.datasets[self.hparams.train_name]
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=self.hparams.dataloader_num_workers,
            # sampler=train_sampler,
            batch_size=self.hparams.batch_size,
            collate_fn=self.pad_batch_collate,
        )

        self.train_dataloader_object = train_dataloader
        return train_dataloader

    def val_dataloader(self):
        """Create dataloader for validation."""
        valid_dataset = self.datasets[self.hparams.val_name]
        valid_dataloader = DataLoader(
            valid_dataset,
            num_workers=self.hparams.dataloader_num_workers,
            # sampler=valid_sampler,
            batch_size=self.hparams.batch_size,
            collate_fn=self.pad_batch_collate,
        )
        return valid_dataloader

    def test_dataloader(self):
        """Create dataloader for testing."""
        self.rouge_metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.rouge_metrics, use_stemmer=True
        )
        test_dataset = self.datasets[self.hparams.test_name]
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=self.hparams.dataloader_num_workers,
            # sampler=test_sampler,
            batch_size=self.hparams.batch_size,
            collate_fn=self.pad_batch_collate,
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

    def training_step(self, batch, batch_idx):  # skipcq: PYL-W0613
        """Training step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.training_step>`__"""  # noqa: E501
        # Get batch information
        labels = batch["labels"]

        # delete labels so now batch contains everything to be inputted into the model
        del batch["labels"]

        # If global_step has increased by 1:
        # Begin training the `word_embedding_model` after `num_frozen_steps` steps
        if (self.global_step_tracker + 1) == self.trainer.global_step:
            self.global_step_tracker = self.trainer.global_step

            if self.emd_model_frozen and (
                self.trainer.global_step > self.hparams.num_frozen_steps
            ):
                self.emd_model_frozen = False
                self.unfreeze_web_model()

        # Compute model forward
        outputs, mask = self.forward(**batch)

        # Compute loss
        (
            loss_total,
            loss_total_norm_batch,
            loss_avg_seq_sum,
            loss_avg_seq_mean,
            loss_avg,
        ) = self.compute_loss(outputs, labels, mask)

        # Generate logs
        loss_dict = {
            "train_loss_total": loss_total,
            "train_loss_total_norm_batch": loss_total_norm_batch,
            "train_loss_avg_seq_sum": loss_avg_seq_sum,
            "train_loss_avg_seq_mean": loss_avg_seq_mean,
            "train_loss_avg": loss_avg,
        }

        for name, value in loss_dict.items():
            self.log(name, value, prog_bar=True, sync_dist=True)

        return loss_dict["train_" + self.hparams.loss_key]

    def validation_step(self, batch, batch_idx):  # skipcq: PYL-W0613
        """
        Validation step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_step>`__
        Similar to :meth:`~extractive.ExtractiveSummarizer.training_step` in that in runs the
        inputs through the model. However, this method also calculates accuracy and f1 score by
        marking every sentence score >0.5 as 1 (meaning should be in the summary) and each score
        <0.5 as 0 (meaning should not be in the summary).
        """  # noqa: E501
        # Get batch information
        labels = batch["labels"]

        # delete labels so now batch contains everything to be inputted into the model
        del batch["labels"]

        # Compute model forward
        outputs, mask = self.forward(**batch)

        # Compute loss
        (
            loss_total,
            loss_total_norm_batch,
            loss_avg_seq_sum,
            loss_avg_seq_mean,
            loss_avg,
        ) = self.compute_loss(outputs, labels, mask)

        # Compute accuracy metrics
        y_hat = torch.sigmoid(outputs)
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        y_hat = torch.flatten(y_hat)
        y_true = torch.flatten(labels)
        result = acc_and_f1(
            y_hat.detach().cpu().numpy(), y_true.float().detach().cpu().numpy()
        )
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
        """
        Called at the end of a validation epoch: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.validation_epoch_end>`__
        Finds the mean of all the metrics logged by :meth:`~extractive.ExtractiveSummarizer.validation_step`.
        """  # noqa: E501
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
        loss_dict = {
            "val_loss_total": avg_loss_total,
            "val_loss_total_norm_batch": avg_loss_total_norm_batch,
            "val_loss_avg_seq_sum": avg_loss_avg_seq_sum,
            "val_loss_avg_seq_mean": avg_loss_avg_seq_mean,
            "val_loss_avg": avg_loss_avg,
            "val_acc": avg_val_acc,
            "val_f1": avg_val_f1,
            "val_acc_and_f1": avg_val_acc_and_f1,
        }

        for name, value in loss_dict.items():
            self.log(name, value, prog_bar=True, sync_dist=True)

        self.log("val_loss", loss_dict["val_" + self.hparams.loss_key], sync_dist=True)

    def test_step(self, batch, batch_idx):
        """
        Test step: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_step>`__
        Similar to :meth:`~extractive.ExtractiveSummarizer.validation_step` in that in runs the
        inputs through the model. However, this method also calculates the ROUGE scores for each
        example-summary pair.
        """  # noqa: E501
        # Get batch information
        labels = batch["labels"]
        sources = batch["source"]
        targets = batch["target"]

        # delete labels, sources, and targets so now batch contains everything to be inputted into
        # the model
        del batch["labels"]
        del batch["source"]
        del batch["target"]

        # Compute model forward
        outputs, _ = self.forward(**batch)
        outputs = torch.sigmoid(outputs)

        # Compute accuracy metrics
        y_hat = outputs.clone().detach()
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        y_hat = torch.flatten(y_hat)
        y_true = torch.flatten(labels)
        result = acc_and_f1(
            y_hat.detach().cpu().numpy(), y_true.float().detach().cpu().numpy()
        )
        acc = torch.tensor(result["acc"])
        f1 = torch.tensor(result["f1"])
        acc_f1 = torch.tensor(result["acc_and_f1"])

        sorted_ids = (
            torch.argsort(outputs, dim=1, descending=True).detach().cpu().numpy()
        )
        if self.hparams.test_id_method == "top_k":
            selected_ids = sorted_ids  # [:, : self.hparams.test_k]
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
                if (index not in (previous_index, previous_index + 1)) or value == -1:
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
                "%s is not a valid option for `--test_id_method`.",
                self.hparams.test_id_method,
            )

        rouge_outputs = []
        predictions = []
        # get ROUGE scores for each (source, target) pair
        for idx, (source, source_ids, target) in enumerate(
            zip(sources, selected_ids, targets)
        ):
            current_prediction = []
            for sent_idx, i in enumerate(source_ids):
                if i >= len(source):
                    logger.debug(
                        "Only %i examples selected from document %i in batch %i. This is likely "
                        + "because some sentences received ranks so small they rounded to zero "
                        + "and a padding 'sentence' was randomly chosen.",
                        sent_idx + 1,
                        idx,
                        batch_idx,
                    )
                    continue

                candidate = source[i].strip()
                # If trigram blocking is enabled and searching for matching trigrams finds no
                # matches then add the candidate to the current prediction list.
                # During the predicting process, Trigram Blocking is used to reduce redundancy.
                # Given selected summary S and a candidate sentence c, we will skip c is there
                # exists a trigram overlapping between c and S.
                if (not self.hparams.no_test_block_trigrams) and (
                    not block_trigrams(candidate, current_prediction)
                ):
                    current_prediction.append(candidate)
                if self.hparams.no_test_block_trigrams:
                    current_prediction.append(candidate)

                # If the testing method is "top_k" and correct number of sentences have been
                # added then break the loop and stop adding sentences. If the testing method
                # is "greater_k" then we will continue to add all the sentences from `selected_ids`
                if (self.hparams.test_id_method == "top_k") and (
                    len(current_prediction) >= self.hparams.test_k
                ):
                    break

            # See this issue https://github.com/google-research/google-research/issues/168
            # for info about the differences between `pyrouge` and `rouge-score`.
            # Archive Link: https://web.archive.org/web/20200622205503/https://github.com/google-research/google-research/issues/168  # noqa: E501
            if self.hparams.test_use_pyrouge:
                # Convert `current_prediction` from list to string with a "<q>" between each
                # item/sentence. In ROUGE 1.5.5 (`pyrouge`), a "\n" indicates sentence
                # boundaries but the below "save_gold.txt" and "save_pred.txt" could not be
                # created if each sentence had to be separated by a newline. Thus, each
                # sentence is separated by a "<q>" token and is then converted to a newline
                # in `helpers.test_rouge`.
                current_prediction = "<q>".join(current_prediction)
                predictions.append(current_prediction)
            else:
                # Convert `current_prediction` from list to string with a newline between each
                # item/sentence. `rouge-score` splits sentences by newline.
                current_prediction = "\n".join(current_prediction)
                target = target.replace("<q>", "\n")
                rouge_outputs.append(
                    self.rouge_scorer.score(target, current_prediction)
                )

        if self.hparams.test_use_pyrouge:
            with open("save_gold.txt", "a+") as save_gold, open(
                "save_pred.txt", "a+"
            ) as save_pred:
                for i in enumerate(targets):
                    save_gold.write(targets[i].strip() + "\n")
                for i in enumerate(predictions):
                    save_pred.write(predictions[i].strip() + "\n")

        output = OrderedDict(
            {
                "test_acc": acc,
                "test_f1": f1,
                "test_acc_and_f1": acc_f1,
                "rouge_scores": rouge_outputs,
            }
        )
        return output

    def test_epoch_end(self, outputs):
        """
        Called at the end of a testing epoch: `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule.test_epoch_end>`__
        Finds the mean of all the metrics logged by :meth:`~extractive.ExtractiveSummarizer.test_step`.
        """  # noqa: E501
        # Get the accuracy metrics over all testing runs
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        avg_test_f1 = torch.stack([x["test_f1"] for x in outputs]).mean()
        avg_test_acc_and_f1 = torch.stack(
            [x["test_acc_and_f1"] for x in outputs]
        ).mean()

        rouge_scores_log = {}

        if self.hparams.test_use_pyrouge:
            test_rouge("tmp", "save_pred.txt", "save_gold.txt")
        else:
            aggregator = scoring.BootstrapAggregator()

            # In `outputs` there is an entry for each batch that was passwed through the
            # `test_step()` function. For each batch a list containing the rouge scores
            # for each example exists under the key "rouge_scores" in `batch_list`. Thus,
            # the below list comprehension loops through the list of outputs and grabs the
            # items stored under the "rouge_scores" key. Then it flattens the list of lists
            # to a list of rouge score objects that can be added to the `aggregator`.
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
            # For more info see the source code: https://github.com/google-research/google-research/blob/master/rouge/scoring.py  # noqa: E501
            rouge_result = aggregator.aggregate()

            for metric, value in rouge_result.items():
                rouge_scores_log[metric + "-precision"] = value.mid.precision
                rouge_scores_log[metric + "-recall"] = value.mid.recall
                rouge_scores_log[metric + "-fmeasure"] = value.mid.fmeasure

        # Generate logs
        loss_dict = {
            "test_acc": avg_test_acc,
            "test_f1": avg_test_f1,
            "avg_test_acc_and_f1": avg_test_acc_and_f1,
        }

        for name, value in loss_dict.items():
            self.log(name, value, prog_bar=True, sync_dist=True)
        for name, value in rouge_scores_log.items():
            self.log(name, value, prog_bar=False, sync_dist=True)

    def predict_sentences(
        self,
        input_sentences: Union[List[str], types.GeneratorType],
        raw_scores=False,
        num_summary_sentences=3,
        tokenized=False,
    ):
        """Summarizes ``input_sentences`` using the model.

        Args:
            input_sentences (list or generator): The sentences to be summarized as a
                list or a generator of spacy Spans (``spacy.tokens.span.Span``), which
                can be obtained by running ``nlp("input document").sents`` where
                ``nlp`` is a spacy model with a sentencizer.
            raw_scores (bool, optional): Return a list containing each sentence
                and its corespoding score instead of the summary. Defaults to False.
            num_summary_sentences (int, optional): The number of sentences in the
                output summary. This value specifies the number of top sentences to
                select as the summary. Defaults to 3.
            tokenized (bool, optional): If the input sentences are already tokenized
                using spacy. If true, ``input_sentences`` should be a list of lists
                where the outer list contains sentences and the inner lists contain
                tokens. Defaults to False.

        Returns:
            str: The summary text. If ``raw_scores`` is set then returns a list
            of input sentences and their corespoding scores.
        """
        # Create source text.
        # Don't add periods when joining because that creates a space before the period.
        if tokenized:
            src_txt = [
                " ".join([token.text for token in sentence if str(token) != "."]) + "."
                for sentence in input_sentences
            ]
        else:
            nlp = English()
            sentencizer = nlp.create_pipe("sentencizer")
            try:
                nlp.add_pipe(sentencizer)
            except ValueError as e:
                if e.args[0].startswith("[E966]"):
                    nlp.add_pipe("sentencizer")
                else:
                    raise e


            src_txt = [
                " ".join([token.text for token in nlp(sentence) if str(token) != "."])
                + "."
                for sentence in input_sentences
            ]

        input_ids = SentencesProcessor.get_input_ids(
            self.tokenizer,
            src_txt,
            sep_token=self.tokenizer.sep_token,
            cls_token=self.tokenizer.cls_token,
            bert_compatible_cls=True,
        )

        input_ids = torch.tensor(input_ids)
        attention_mask = [1] * len(input_ids)
        attention_mask = torch.tensor(attention_mask)

        sent_rep_token_ids = [
            i for i, t in enumerate(input_ids) if t == self.tokenizer.cls_token_id
        ]
        sent_rep_mask = torch.tensor([1] * len(sent_rep_token_ids))

        input_ids.unsqueeze_(0)
        attention_mask.unsqueeze_(0)
        sent_rep_mask.unsqueeze_(0)

        self.eval()

        with torch.no_grad():
            outputs, _ = self.forward(
                input_ids,
                attention_mask,
                sent_rep_mask=sent_rep_mask,
                sent_rep_token_ids=sent_rep_token_ids,
            )
            outputs = torch.sigmoid(outputs)

        if raw_scores:
            # key=sentence
            # value=score
            sent_scores = list(zip(src_txt, outputs.tolist()[0]))
            return sent_scores

        sorted_ids = (
            torch.argsort(outputs, dim=1, descending=True).detach().cpu().numpy()
        )
        logger.debug("Sorted sentence ids: %s", sorted_ids)
        selected_ids = sorted_ids[0, :num_summary_sentences]
        logger.debug("Selected sentence ids: %s", selected_ids)

        selected_sents = []
        selected_ids.sort()
        for i in selected_ids:
            selected_sents.append(src_txt[i])

        return " ".join(selected_sents).strip()

    def predict(self, input_text: str, raw_scores=False, num_summary_sentences=3):
        """Summarizes ``input_text`` using the model.

        Args:
            input_text (str): The text to be summarized.
            raw_scores (bool, optional): Return a list containing each sentence
                and its corespoding score instead of the summary. Defaults to False.
            num_summary_sentences (int, optional): The number of sentences in the
                output summary. This value specifies the number of top sentences to
                select as the summary. Defaults to 3.

        Returns:
            str: The summary text. If ``raw_scores`` is set then returns a list
            of input sentences and their corespoding scores.
        """
        nlp = English()
        nlp.add_pipe("sentencizer")
        doc = nlp(input_text)

        return self.predict_sentences(
            input_sentences=doc.sents,
            raw_scores=raw_scores,
            num_summary_sentences=num_summary_sentences,
            tokenized=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Arguments specific to this model"""
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="bert-base-uncased",
            help="Path to pre-trained model or shortcut name. A list of shortcut names can be "
            + "found at https://huggingface.co/transformers/pretrained_models.html. "
            + "Community-uploaded models are located at https://huggingface.co/models.",
        )
        parser.add_argument(
            "--model_type",
            type=str,
            default="bert",
            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES),
        )
        parser.add_argument("--tokenizer_name", type=str, default="")
        parser.add_argument(
            "--tokenizer_no_use_fast",
            action="store_true",
            help="Don't use the fast version of the tokenizer for the specified model. "
            + "More info: https://huggingface.co/transformers/main_classes/tokenizer.html.",
        )
        parser.add_argument(
            "--max_seq_length",
            type=int,
            default=0,
            help="The maximum sequence length of processed documents.",
        )
        parser.add_argument(
            "--data_path", type=str, help="Directory containing the dataset."
        )
        parser.add_argument(
            "--data_type",
            default="none",
            type=str,
            choices=["txt", "pt", "none"],
            help="""The file extension of the prepared data. The 'map' `--dataloader_type`
            requires `txt` and the 'iterable' `--dataloader_type` works with both. If the data
            is not prepared yet (in JSON format) this value specifies the output format
            after processing. If the data is prepared, this value specifies the format to load.
            If it is `none` then the type of data to be loaded will be inferred from the
            `data_path`. If data needs to be prepared, this cannot be set to `none`.""",
        )
        parser.add_argument("--num_threads", type=int, default=4)
        parser.add_argument("--processing_num_threads", type=int, default=2)
        parser.add_argument(
            "--pooling_mode",
            type=str,
            default="sent_rep_tokens",
            choices=["sent_rep_tokens", "mean_tokens", "max_tokens"],
            help="How word vectors should be converted to sentence embeddings.",
        )
        parser.add_argument(
            "--num_frozen_steps",
            type=int,
            default=0,
            help="Freeze (don't train) the word embedding model for this many steps.",
        )
        parser.add_argument(
            "--batch_size",
            default=8,
            type=int,
            help="Batch size per GPU/CPU for training/evaluation/testing.",
        )
        parser.add_argument(
            "--dataloader_type",
            default="map",
            type=str,
            choices=["map", "iterable"],
            help="The style of dataloader to use. `map` is faster and uses less memory.",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            default=4,
            type=int,
            help="""The number of workers to use when loading data. A general place to
            start is to set num_workers equal to the number of CPU cores on your machine.
            If `--dataloader_type` is 'iterable' then this setting has no effect and
        num_workers will be 1. More details here: https://pytorch-lightning.readthedocs.io/en/latest/performance.html#num-workers""",  # noqa: E501
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
            "--preprocess_resume",
            action="store_true",
            help="Resume preprocessing. `--only_preprocess` must be set in order to resume. "
            + "Determines which files to process by finding the shards that do not have a "
            + 'coresponding ".pt" file in the data directory.',
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
            "--classifier",
            type=str,
            choices=["linear", "simple_linear", "transformer", "transformer_linear"],
            default="simple_linear",
            help="""Which classifier/encoder to use to reduce the hidden dimension of the sentence vectors.
                    `linear` - a `LinearClassifier` with two linear layers, dropout, and an activation function.
                    `simple_linear` - a `LinearClassifier` with one linear layer and a sigmoid.
                    `transformer` - a `TransformerEncoderClassifier` which runs the sentence vectors through some
                                    `nn.TransformerEncoderLayer`s and then a simple `nn.Linear` layer.
                    `transformer_linear` - a `TransformerEncoderClassifier` with a `LinearClassifier` as the
                                           `reduction` parameter, which results in the same thing as the `transformer` option but with a
                                           `LinearClassifier` instead of a `nn.Linear` layer.""",  # noqa: E501
        )
        parser.add_argument(
            "--classifier_dropout",
            type=float,
            default=0.1,
            help="The value for the dropout layers in the classifier.",
        )
        parser.add_argument(
            "--classifier_transformer_num_layers",
            type=int,
            default=2,
            help="The number of layers for the `transformer` classifier. Only has an effect if "
            + '`--classifier` contains "transformer".',
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
            default="top_k",
            choices=["greater_k", "top_k"],
            help="How to chose the top predictions from the model for ROUGE scores.",
        )
        parser.add_argument(
            "--test_k",
            type=float,
            default=3,
            help="The `k` parameter for the `--test_id_method`. Must be set if using the "
            + "`greater_k` option. (default: 3)",
        )
        parser.add_argument(
            "--no_test_block_trigrams",
            action="store_true",
            help="Disable trigram blocking when calculating ROUGE scores during testing. "
            + "This will increase repetition and thus decrease accuracy.",
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
            "--loss_key",
            type=str,
            choices=[
                "loss_total",
                "loss_total_norm_batch",
                "loss_avg_seq_sum",
                "loss_avg_seq_mean",
                "loss_avg",
            ],
            default="loss_avg_seq_mean",
            help="Which reduction method to use with BCELoss. See the "
            + "`experiments/loss_functions/` folder for info on how the default "
            + "(`loss_avg_seq_mean`) was chosen.",
        )
        return parser
