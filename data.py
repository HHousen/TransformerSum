import copy
import csv
import json
import logging

logger = logging.getLogger(__name__)

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

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, sent_rep_token_ids=None):
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

class SentencesProcessor():
    def __init__(self, name=None, labels=None, examples=None, verbose=False):
        self.name = name
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    @classmethod
    def create_from_csv(
        cls, file_name, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
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
            texts, labels, ids, overwrite_labels=overwrite_labels, overwrite_examples=overwrite_examples
        )

    def add_examples(
        self, texts, labels=None, ids=None, oracle_ids=None, overwrite_labels=False, overwrite_examples=False
    ):
        assert labels is None or len(texts) == len(labels)
        assert ids is None or len(texts) == len(ids)
        assert not (labels and oracle_ids)
        assert isinstance(text, list)

        if ids is None:
            ids = [None] * len(texts)
        if labels is None:
            if oracle_ids: # convert `oracle_ids` to `labels`
                labels = [0] * len(src)
                for l in oracle_ids:
                    labels[l] = 1
            else:
                labels = [None] * len(texts)
        
        examples = []
        added_labels = list()
        for (text_set, label_set, guid) in zip(texts, labels, ids):
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

    def preprocess_examples(self, min_sentence_ntokens=5, max_sentence_ntokens=200, min_example_nsents=3, max_example_nsents=100):
        for (ex_index, example) in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logger.info("Preprocessing example %d", ex_index)
            # pick the sentence indexes in `example.text` if they are larger then `min_sentence_ntokens`
            idxs = [i for i, s in enumerate(example.text) if (len(s) > min_sentence_ntokens)]
            # truncate selected source sentences to `max_sentence_ntokens`
            example.text = [example.text[i][:max_sentence_ntokens] for i in idxs]
            # only pick labels for sentences that matched the length requirement
            example.labels = [example.labels[i] for i in idxs]
            # truncate entire source to max number of sentences (`max_example_nsents`)
            example.text = example.text[:max_example_nsents]
            # perform above truncation to `labels`
            example.labels = example.labels[:max_example_nsents]

            # if the example does not meet the length requirement then remove it
            if (len(example.text) < min_example_nsents):
                self.examples.pop(ex_index)
                self.labels.pop(ex_index)

    def get_features(
        self,
        tokenizer,
        bert_compatible_cls=True,
        create_sent_rep_token_ids=True,
        sent_rep_token_id=None,
        create_segment_ids=True,
        segment_token_id=None,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=False,
        save_to_path=None,
        save_to_name=None
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

        batch_length = max(len(input_ids) for input_ids in all_input_ids)

        if create_sent_rep_token_ids:
            if sent_rep_token_id == "sep": # get the sep token id
                sent_rep_token_id = tokenizer.sep_token_id
            elif sent_rep_token_id == "cls": # get the cls token id
                sent_rep_token_id = tokenizer.cls_token_id
            else: # default to trying to get the `sep_token_id` if the `sent_rep_token_id` is not set
                sent_rep_token_id = tokenizer.sep_token_id
        
        if create_segment_ids:
            if segment_token_id == "period": # get the token id for a "."
                segment_token_id = tokenizer.convert_tokens_to_ids(["."])[0]
            else: # default to trying to get the `cls_token_id` if the `segment_token_id` is not 
                segment_token_id = tokenizer.cls_token_id

        for (ex_index, (example, label)) in enumerate(zip(self.examples, self.labels)):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)
            
            if bert_compatible_cls: # adds a '[CLS]' token between each sentence and outputs `input_ids`
                # convert `example.text` to array of sentences
                src_txt = [' '.join(sent) for sent in example.text]
                # separate each sentence with ' [SEP] [CLS] ' and convert to string 
                text = ' [SEP] [CLS] '.join(src_txt)
                # tokenize
                src_subtokens = self.tokenizer.tokenize(text)
                # select first `(max_length-2)` tokens (so the following line of tokens can be added)
                src_subtokens = src_subtokens[:(max_length-2)]
                # add '[CLS]' to beginning and '[SEP]' to end
                src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
                # create `input_ids`
                input_ids = self.tokenizer.convert_tokens_to_ids(src_subtokens)
            else:
                input_ids = tokenizer.encode(
                    example.text, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len),
                )

            # Segment (Token Type) IDs
            segment_ids = None
            if create_segment_ids:
                current_segment = 1
                segment_ids = []
                for token in input_ids:
                    if token == segment_token_id:
                        current_segment += 1
                    segment_ids += [current_segment]
            
            # Sentence Representation Token IDs
            sent_rep_ids = None
            if create_sent_rep_token_ids:
                # create list of indexes for the `sent_rep` tokens
                sent_rep_ids = [i for i, t in enumerate(src_subtoken_ids) if t == sent_rep_token_id]
                # truncate `label` to the length of the `cls_ids` aka the number of sentences
                label = label[:len(sent_rep_ids)]
            
            # Attention
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Padding
            # Zero-pad up to the sequence length.
            padding_length = batch_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_ids) == batch_length, "Error with input length {} vs {}".format(
                len(input_ids), batch_length
            )
            assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
                len(attention_mask), batch_length
            )

            if ex_index < 5 and self.verbose:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                if segment_ids is not None:
                    logger.info("token_type_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if sent_rep_ids is not None:
                    logger.info("sent_rep_token_ids: %s" % " ".join([str(x) for x in sent_rep_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("labels: %s (id = %s)" % (example.labels, label))

            # Append all features
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label,
                    token_type_ids=segment_ids,
                    sent_rep_token_ids=sent_rep_ids
                )
            )

        if return_tensors is False:
            return features
        else:
            import torch
            from torch.utils.data import TensorDataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_sent_rep_token_ids = torch.tensor([f.all_sent_rep_token_ids for f in features], dtype=torch.long)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_token_type_ids, all_sent_rep_token_ids)
            
            if save_to_path:
                dataset_path = os.path.join(
                    save_to_path,
                    (save_to_name if save_to_name else "dataset"),
                )
                logger.info("Saving dataset into cached file %s", dataset_path)
                torch.save(dataset, dataset_path)
            
            return dataset

    def load(self, load_from_path, dataset_name=None):
        """ Attempts to load the dataset from storage. If that fails, will return None. """
        dataset_path = os.path.join(
            load_from_path,
            (dataset_name if dataset_name else ("dataset_" + self.name)),
        )
        if os.path.exists(dataset_path):
            logger.info("Loading data from file %s", dataset_path)
            dataset = torch.load(dataset_path)
            return dataset
        else:
            return None

# Section Methods (to convert abstractive summary to extractive)
def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)
