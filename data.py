import os
import re
import gc
import copy
import csv
import json
import logging
import torch
from tqdm import tqdm
from time import time
from multiprocessing import Pool
from functools import partial
import torch
from torch.utils.data import TensorDataset, IterableDataset

logger = logging.getLogger(__name__)


class FSTensorDataset(IterableDataset):
    def __init__(self, files_list):
        super(FSTensorDataset).__init__()
        self.files_list = files_list

    def __iter__(self):
        for data_file in self.files_list:
            dataset_section = torch.load(data_file)
            for example in dataset_section:
                yield example
            # Clear memory usage before loading next file
            dataset_section = None
            gc.collect()
            del dataset_section
            gc.collect()


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text, labels):
        self.guid = guid
        self.text = text
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        sent_rep_token_ids=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.sent_rep_token_ids = sent_rep_token_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentencesProcessor:
    def __init__(self, name=None, labels=None, examples=None, verbose=False):
        self.name = name
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    @classmethod
    def create_from_csv(
        cls,
        file_name,
        split_name="",
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        **kwargs
    ):
        processor = cls(**kwargs)
        processor.add_examples_from_csv(
            file_name,
            split_name=split_name,
            column_label=column_label,
            column_text=column_text,
            column_id=column_id,
            skip_first_row=skip_first_row,
            overwrite_labels=True,
            overwrite_examples=True,
        )
        return processor

    @classmethod
    def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
        processor = cls(**kwargs)
        processor.add_examples(texts_or_text_and_labels, labels=labels)
        return processor

    def add_examples_from_csv(
        self,
        file_name,
        split_name="",
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        lines = self._read_tsv(file_name)
        if skip_first_row:
            lines = lines[1:]
        texts = []
        labels = []
        ids = []
        for (i, line) in enumerate(lines):
            texts.append(line[column_text])
            labels.append(line[column_label])
            if column_id is not None:
                ids.append(line[column_id])
            else:
                guid = "%s-%s" % (split_name, i) if split_name else "%s" % i
                ids.append(guid)

        return self.add_examples(
            texts,
            labels,
            ids,
            overwrite_labels=overwrite_labels,
            overwrite_examples=overwrite_examples,
        )

    def add_examples(
        self,
        texts,
        labels=None,
        ids=None,
        oracle_ids=None,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        assert texts  # not an empty array
        assert labels is None or len(texts) == len(labels)
        assert ids is None or len(texts) == len(ids)
        assert not (labels and oracle_ids)
        assert isinstance(texts, list)

        if ids is None:
            ids = [None] * len(texts)
        if labels is None:
            if oracle_ids:  # convert `oracle_ids` to `labels`
                labels = []
                for text_set, oracle_id in zip(texts, oracle_ids):
                    text_label = [0] * len(text_set)
                    for l in oracle_id:
                        text_label[l] = 1
                    labels.append(text_label)
            else:
                labels = [None] * len(texts)

        examples = []
        added_labels = list()
        for (text_set, label_set, guid) in zip(texts, labels, ids):
            if not text_set or not label_set:
                continue  # input()
            added_labels.append(label_set)
            examples.append(InputExample(guid=guid, text=text_set, labels=label_set))

        # Update examples
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # Update labels
        if overwrite_labels:
            self.labels = added_labels
        else:
            self.labels += added_labels

        return self.examples

    # def preprocess_examples(self, examples, labels, min_sentence_ntokens=5, max_sentence_ntokens=200, min_example_nsents=3, max_example_nsents=100):
    #     for (ex_index, example) in enumerate(examples):
    #         if ex_index % 10000 == 0:
    #             logger.info("Preprocessing example %d", ex_index)
    #         # pick the sentence indexes in `example.text` if they are larger then `min_sentence_ntokens`
    #         idxs = [i for i, s in enumerate(example.text) if (len(s) > min_sentence_ntokens)]
    #         # truncate selected source sentences to `max_sentence_ntokens`
    #         example.text = [example.text[i][:max_sentence_ntokens] for i in idxs]
    #         # only pick labels for sentences that matched the length requirement
    #         example.labels = [example.labels[i] for i in idxs]
    #         # truncate entire source to max number of sentences (`max_example_nsents`)
    #         example.text = example.text[:max_example_nsents]
    #         # perform above truncation to `labels`
    #         example.labels = example.labels[:max_example_nsents]

    #         # if the example does not meet the length requirement then remove it
    #         if (len(example.text) < min_example_nsents):
    #             examples.pop(ex_index)
    #             labels.pop(ex_index)
    #         return examples, labels

    def get_features_process(
        self,
        features,
        input_information,
        num_examples=0,
        tokenizer=None,
        bert_compatible_cls=True,
        create_sent_rep_token_ids=True,
        sent_rep_token_id=None,
        create_segment_ids="binary",
        segment_token_id=None,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ):
        ex_index, example, label = input_information
        if ex_index % 1000 == 0:
            logger.info(
                "Generating features for example "
                + str(ex_index)
                + "/"
                + str(num_examples)
            )
        if (
            bert_compatible_cls
        ):  # adds a '[CLS]' token between each sentence and outputs `input_ids`
            # convert `example.text` to array of sentences
            src_txt = [" ".join(sent) for sent in example.text]
            # separate each sentence with ' [SEP] [CLS] ' and convert to string
            text = " [SEP] [CLS] ".join(src_txt)
            # tokenize
            src_subtokens = tokenizer.tokenize(text)
            # select first `(max_length-2)` tokens (so the following line of tokens can be added)
            src_subtokens = src_subtokens[: (max_length - 2)]
            # add '[CLS]' to beginning and '[SEP]' to end
            src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
            # create `input_ids`
            input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
        else:
            input_ids = tokenizer.encode(
                example.text,
                add_special_tokens=True,
                max_length=min(max_length, tokenizer.max_len),
            )

        # Segment (Token Type) IDs
        segment_ids = None
        if create_segment_ids == "binary":
            current_segment_flag = True
            segment_ids = []
            for token in input_ids:
                if token == segment_token_id:
                    current_segment_flag = not current_segment_flag
                segment_ids += [0 if current_segment_flag else 1]

        if create_segment_ids == "sequential":
            current_segment = 0
            segment_ids = []
            for token in input_ids:
                if token == segment_token_id:
                    current_segment += 1
                segment_ids += [current_segment]

        # Sentence Representation Token IDs
        sent_rep_ids = None
        if create_sent_rep_token_ids:
            # create list of indexes for the `sent_rep` tokens
            sent_rep_ids = [
                i for i, t in enumerate(input_ids) if t == sent_rep_token_id
            ]
            # truncate `label` to the length of the `cls_ids` aka the number of sentences
            label = label[: len(sent_rep_ids)]

        # Attention
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Padding
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + attention_mask
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert (
            len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)

        if ex_index < 5 and self.verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            if segment_ids is not None:
                logger.info(
                    "token_type_ids: %s" % " ".join([str(x) for x in segment_ids])
                )
            if sent_rep_ids is not None:
                logger.info(
                    "sent_rep_token_ids: %s" % " ".join([str(x) for x in sent_rep_ids])
                )
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )
            logger.info("labels: %s (id = %s)" % (example.labels, label))

        # Return features
        return InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label,
            token_type_ids=segment_ids,
            sent_rep_token_ids=sent_rep_ids,
        )

    def get_features(
        self,
        tokenizer,
        bert_compatible_cls=True,
        create_sent_rep_token_ids=True,
        sent_rep_token_id=None,
        create_segment_ids="binary",
        segment_token_id=None,
        n_process=2,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=False,
        save_to_path=None,
        save_to_name=None,
    ):
        """
        Convert examples in a list of ``InputFeatures``

        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """
        if max_length is None:
            max_length = tokenizer.max_len

        # batch_length = max(len(input_ids) for input_ids in all_input_ids)

        if create_sent_rep_token_ids:
            if sent_rep_token_id == "sep":  # get the sep token id
                sent_rep_token_id = tokenizer.sep_token_id
            elif sent_rep_token_id == "cls":  # get the cls token id
                sent_rep_token_id = tokenizer.cls_token_id
            else:  # default to trying to get the `sep_token_id` if the `sent_rep_token_id` is not set
                sent_rep_token_id = tokenizer.sep_token_id

        if create_segment_ids:
            if segment_token_id == "period":  # get the token id for a "."
                segment_token_id = tokenizer.convert_tokens_to_ids(["."])[0]
            else:  # default to trying to get the `cls_token_id` if the `segment_token_id` is not
                segment_token_id = tokenizer.cls_token_id

        features = []
        pool = Pool(n_process)
        _get_features_process = partial(
            self.get_features_process,
            features,
            num_examples=len(self.labels),
            tokenizer=tokenizer,
            bert_compatible_cls=bert_compatible_cls,
            create_sent_rep_token_ids=create_sent_rep_token_ids,
            sent_rep_token_id=sent_rep_token_id,
            create_segment_ids=create_segment_ids,
            segment_token_id=segment_token_id,
            max_length=max_length,
            pad_on_left=pad_on_left,
            pad_token=pad_token,
            mask_padding_with_zero=mask_padding_with_zero,
        )

        for rtn_features in pool.map(
            _get_features_process,
            zip(range(len(self.labels)), self.examples, self.labels),
        ):
            features.append(rtn_features)

        pool.close()
        pool.join()

        # for (ex_index, (example, label)) in tqdm(enumerate(zip(self.examples, self.labels)), total=len(self.labels), desc="Creating Features"):
        #     # if ex_index % 1000 == 0:
        #     #     logger.info("Tokenizing example %d", ex_index)

        #     if bert_compatible_cls: # adds a '[CLS]' token between each sentence and outputs `input_ids`
        #         # convert `example.text` to array of sentences
        #         src_txt = [' '.join(sent) for sent in example.text]
        #         # separate each sentence with ' [SEP] [CLS] ' and convert to string
        #         text = ' [SEP] [CLS] '.join(src_txt)
        #         # tokenize
        #         src_subtokens = tokenizer.tokenize(text)
        #         # select first `(max_length-2)` tokens (so the following line of tokens can be added)
        #         src_subtokens = src_subtokens[:(max_length-2)]
        #         # add '[CLS]' to beginning and '[SEP]' to end
        #         src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        #         # create `input_ids`
        #         input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
        #     else:
        #         input_ids = tokenizer.encode(
        #             example.text, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len),
        #         )

        #     # Segment (Token Type) IDs
        #     segment_ids = None
        #     if create_segment_ids == "binary":
        #         current_segment_flag = True
        #         segment_ids = []
        #         for token in input_ids:
        #             if token == segment_token_id:
        #                 current_segment_flag = not current_segment_flag
        #             segment_ids += [0 if current_segment_flag else 1]

        #     if create_segment_ids == "sequential":
        #         current_segment = 0
        #         segment_ids = []
        #         for token in input_ids:
        #             if token == segment_token_id:
        #                 current_segment += 1
        #             segment_ids += [current_segment]

        #     # Sentence Representation Token IDs
        #     sent_rep_ids = None
        #     if create_sent_rep_token_ids:
        #         # create list of indexes for the `sent_rep` tokens
        #         sent_rep_ids = [i for i, t in enumerate(input_ids) if t == sent_rep_token_id]
        #         # truncate `label` to the length of the `cls_ids` aka the number of sentences
        #         label = label[:len(sent_rep_ids)]

        #     # Attention
        #     # The mask has 1 for real tokens and 0 for padding tokens. Only real
        #     # tokens are attended to.
        #     attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        #     # Padding
        #     # Zero-pad up to the sequence length.
        #     padding_length = max_length - len(input_ids)
        #     if pad_on_left:
        #         input_ids = ([pad_token] * padding_length) + input_ids
        #         attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        #     else:
        #         input_ids = input_ids + ([pad_token] * padding_length)
        #         attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        #     assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
        #         len(input_ids), max_length
        #     )
        #     assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        #         len(attention_mask), max_length
        #     )

        #     if ex_index < 5 and self.verbose:
        #         logger.info("*** Example ***")
        #         logger.info("guid: %s" % (example.guid))
        #         logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #         if segment_ids is not None:
        #             logger.info("token_type_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #         if sent_rep_ids is not None:
        #             logger.info("sent_rep_token_ids: %s" % " ".join([str(x) for x in sent_rep_ids]))
        #         logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #         logger.info("labels: %s (id = %s)" % (example.labels, label))

        #     # Append to features
        #     features.append(
        #         InputFeatures(
        #             input_ids=input_ids,
        #             attention_mask=attention_mask,
        #             labels=label,
        #             token_type_ids=segment_ids,
        #             sent_rep_token_ids=sent_rep_ids
        #         )
        #     )

        if return_tensors is False:
            return features
        else:

            def pad(data, pad_id, width=None):
                if not width:
                    width = max(len(d) for d in data)
                rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
                return rtn_data

            # Pad sentence representation token ids (`sent_rep_token_ids`)
            all_sent_rep_token_ids_padded = torch.tensor(
                pad([f.sent_rep_token_ids for f in features], -1), dtype=torch.long
            )
            all_sent_rep_token_ids_masks = ~(all_sent_rep_token_ids_padded == -1)
            all_sent_rep_token_ids_padded[all_sent_rep_token_ids_padded == -1] = 0

            all_input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            )
            all_attention_masks = torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            )
            all_labels = torch.tensor(
                pad([f.labels for f in features], 0), dtype=torch.long
            )
            all_token_type_ids = torch.tensor(
                pad([f.token_type_ids for f in features], 0), dtype=torch.long
            )
            all_sent_rep_token_ids = torch.tensor(
                all_sent_rep_token_ids_padded, dtype=torch.long
            )
            all_sent_rep_token_ids_masks = torch.tensor(
                all_sent_rep_token_ids_masks, dtype=torch.long
            )

            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_labels,
                all_token_type_ids,
                all_sent_rep_token_ids,
                all_sent_rep_token_ids_masks,
            )

            if save_to_path:
                final_save_name = (
                    save_to_name if save_to_name else ("dataset_" + self.name)
                )
                dataset_path = os.path.join(save_to_path, (final_save_name + ".pt"),)
                logger.info("Saving dataset into cached file %s", dataset_path)
                torch.save(dataset, dataset_path)

            return dataset

    def load(self, load_from_path, dataset_name=None):
        """ Attempts to load the dataset from storage. If that fails, will return None. """
        final_load_name = dataset_name if dataset_name else ("dataset_" + self.name)
        dataset_path = os.path.join(load_from_path, (final_load_name + ".pt"),)
        if os.path.exists(dataset_path):
            logger.info("Loading data from file %s", dataset_path)
            dataset = torch.load(dataset_path)
            return dataset
        else:
            return None
