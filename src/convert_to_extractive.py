import gc
import glob
import gzip
import itertools
import json
import logging
import math
import os
import re
import sys
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from time import time

import spacy
from spacy.lang.en import English
from tqdm import tqdm

import datasets as hf_nlp
from helpers import _get_word_ngrams, load_json

logger = logging.getLogger(__name__)

# Steps
# Run cnn/dm processing script to get train, test, valid bin text files
# For each bin:
#   Load all the data where each line is an entry in a list
#   For each document (line) (parallelized):
#       Tokenize line in the source and target files
#       Run source and target through oracle_id algorithm
#       Run current preprocess_examples() function (data cleaning) in data processor
#       Return source (as list of sentences) and target
#   In map() loop: append each (source, target, labels) to variable and save (as
#   cnn_dm_extractive) once done

# BertSum:
# 1. Tokenize all files into tokenized json versions
# 2. Split json into source and target AND concat stories into chunks of `shard_size`
#    number of stories
# 3. Process to obtain extractive summary and labels for each shard
# 4. Save each processed shard as list of dictionaries with processed values


def read_in_chunks(file_object, chunk_size=5000):
    """Read a file line by line but yield chunks of ``chunk_size`` number of lines at a time."""
    # https://stackoverflow.com/a/519653
    # zero mod anything is zero so start counting at 1
    current_line_num = 1
    lines = []
    for line in file_object:
        # use `(chunk_size + 1)` because each iteration starts at 1
        if current_line_num % (chunk_size + 1) == 0:
            yield lines
            # reset the `lines` so a new chunk can be yielded
            lines.clear()
            # Essentially adds one twice so that each interation starts counting at one.
            # This means each yielded chunk will be the same size instead of the first
            # one being 5000 then every one after being 5001, for example.
            current_line_num += 1
        lines.append(line.strip())
        current_line_num += 1
    # When the `file_object` has no more lines left then yield the current chunk,
    # even if it is not a chunk of 5000 (`chunk_size`) as long as it contains more than
    # 0 examples.
    if len(lines) > 0:
        yield lines


def convert_to_extractive_driver(args):
    """
    Driver function to convert an abstractive summarization dataset to an extractive dataset.
    The abstractive dataset must be formatted with two files for each split: a source and target
    file. Example file list for two splits:
    ``["train.source", "train.target", "val.source", "val.target"]``
    """
    # default is to output to input data directory if no output directory specified
    if not args.base_output_path:
        args.base_output_path = args.base_path

    # load spacy english small model with the "tagger" and "ner" disabled since
    # we only need the "tokenizer" and "parser"
    # more info: https://spacy.io/usage/processing-pipelines
    if args.sentencizer:
        nlp = English()
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)
    else:
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])

    if args.dataset:
        dataset = hf_nlp.load_dataset(args.dataset, args.dataset_version)

    # for each split
    for name in tqdm(
        args.split_names, total=len(args.split_names), desc="Dataset Split"
    ):
        if args.dataset:  # if loading using the `nlp` library
            current_dataset = dataset[name]
            source_file = current_dataset[args.data_example_column]
            target_file = current_dataset[args.data_summarized_column]
        else:
            # get the source and target paths
            source_file_path = os.path.join(
                args.base_path, (name + "." + args.source_ext)
            )
            target_file_path = os.path.join(
                args.base_path, (name + "." + args.target_ext)
            )
            logger.info("Opening source and target %s files", name)
            source_file = open(source_file_path, "r")
            target_file = open(target_file_path, "r")

        if args.shard_interval:  # if sharding is enabled
            # get number of examples to process
            if args.dataset:
                target_file_len = len(current_dataset)
            else:
                target_file_len = sum(1 for line in target_file)
                # reset pointer back to beginning after getting length
                target_file.seek(0)

            # find how long the loop will run, round up because any extra examples
            # will form a chunk of size less than `args.shard_interval`
            tot_num_interations = math.ceil(target_file_len / args.shard_interval)

            # default is that there was no previous shard (aka not resuming)
            last_shard = 0
            if args.resume:
                assert (
                    not args.dataset
                ), "Cannot resume when using data loaded from the `nlp` library."
                num_lines_read, last_shard = resume(
                    args.base_output_path, name, args.shard_interval
                )

                # if lines have been read and shards have been written to disk
                if num_lines_read:
                    logger.info("Resuming to line %i", num_lines_read - 1)
                    # seek both the source and target to the next line
                    seek_files([source_file, target_file], num_lines_read - 1)

                    # checks to make sure the last documents match
                    # this moves the file pointer in source_file forward 1...
                    resume_success = check_resume_success(
                        nlp,
                        args,
                        source_file,
                        last_shard,
                        args.base_output_path,
                        name,
                        args.compression,
                    )
                    # ...so move the target_file pointed forward 1 as well
                    target_file.readline()

                    if not resume_success:
                        logger.error("Exiting...")
                        sys.exit(-1)

                    # subtract the number of shards already created
                    tot_num_interations -= int(last_shard)
                else:  # no shards on disk
                    logger.warning("Tried to resume but no shards found on disk")

            for piece_idx, (source_docs, target_docs) in tqdm(
                enumerate(
                    zip(
                        read_in_chunks(source_file, args.shard_interval),
                        read_in_chunks(target_file, args.shard_interval),
                    )
                ),
                total=tot_num_interations,
                desc="Shards",
            ):
                piece_idx += last_shard  # effective if resuming (offsets the index)
                convert_to_extractive_process(
                    args, nlp, source_docs, target_docs, name, piece_idx
                )
        else:
            # only `str.strip()` the lines if loading from an actual file, not
            # the `nlp` library
            if args.dataset:
                source_docs = source_file
                target_docs = target_file
            else:
                source_docs = [line.strip() for line in source_file]
                target_docs = [line.strip() for line in target_file]
            convert_to_extractive_process(args, nlp, source_docs, target_docs, name)

        # If not processing data from the `nlp` library then close the loaded files
        if not args.dataset:
            source_file.close()
            target_file.close()


def convert_to_extractive_process(
    args, nlp, source_docs, target_docs, name, piece_idx=None
):
    """
    Main process to convert an abstractive summarization dataset to extractive.
    Tokenizes, gets the ``oracle_ids``, splits into ``source`` and ``labels``, and
    saves processed data.
    """
    # tokenize the source and target documents
    # each step runs in parallel on `args.n_process` threads with batch size `args.batch_size`

    source_docs_tokenized = tokenize(
        nlp,
        source_docs,
        args.n_process,
        args.batch_size,
        name=(" " + name + "-" + args.source_ext),
        tokenizer_log_interval=args.tokenizer_log_interval,
    )
    del source_docs
    target_docs_tokenized = tokenize(
        nlp,
        target_docs,
        args.n_process,
        args.batch_size,
        name=(" " + name + "-" + args.target_ext),
        tokenizer_log_interval=args.tokenizer_log_interval,
    )
    # set a constant `oracle_mode`
    _example_processor = partial(
        example_processor,
        args=args,
        oracle_mode=args.oracle_mode,
        no_preprocess=args.no_preprocess,
    )

    dataset = []
    pool = Pool(args.n_process)
    logger.info("Processing %s", name)
    t0 = time()
    for (preprocessed_data, target_doc) in pool.map(
        _example_processor,
        zip(source_docs_tokenized, target_docs_tokenized),
    ):
        if preprocessed_data is not None:
            # preprocessed_data is (source_doc, labels)
            to_append = {"src": preprocessed_data[0], "labels": preprocessed_data[1]}
            if name in args.add_target_to:
                # Convert the tokenized list of sentences where each sentence is a list of tokens
                # to a string where each sentence is separated by "<q>". This is necessary for
                # proper ROUGE score calculation. Each sentence should be separated by a newline
                # for ROUGE to work properly, but we use "<q>" and convert it back to a newline
                # when necessary since "<q>" is easier to store than "\n".
                to_append["tgt"] = "<q>".join([" ".join(sent) for sent in target_doc])
            dataset.append(to_append)

    pool.close()
    pool.join()
    del source_docs_tokenized
    del target_docs
    del target_docs_tokenized
    gc.collect()

    logger.info("Done in %.2f seconds", time() - t0)

    if args.shard_interval:
        split_output_path = os.path.join(
            args.base_output_path, (name + "." + str(piece_idx) + ".json")
        )
    else:
        split_output_path = os.path.join(args.base_output_path, (name + ".json"))
    save(dataset, split_output_path, compression=args.compression)

    del dataset
    gc.collect()


def resume(output_path, split, chunk_size):
    """
    Find the last shard created and return the total number of lines read and last
    shard number.
    """
    glob_str = os.path.join(output_path, (split + ".*.json*"))
    all_json_in_split = glob.glob(glob_str)

    if not all_json_in_split:  # if no files found
        return None

    # get the first match because and convert to int so max() operator works
    # more info about the below RegEx: https://stackoverflow.com/a/1454936
    # (https://web.archive.org/web/20200701145857/https://stackoverflow.com/questions/1454913/regular-expression-to-find-a-string-included-between-two-characters-while-exclud/1454936) # noqa: E501
    shard_file_idxs = [
        int(re.search(r"(?<=\.)(.*?)(?=\.)", a).group(1)) for a in all_json_in_split
    ]

    last_shard = int(max(shard_file_idxs)) + 1  # because the file indexes start at 0

    num_lines_read = chunk_size * last_shard
    # `num_lines_read` is the number of lines read if line indexing started at 1
    # therefore, this number is the number of the next line wanted
    return num_lines_read, last_shard


def check_resume_success(
    nlp, args, source_file, last_shard, output_path, split, compression
):
    logger.info("Checking if resume was successful...")
    chunk_file_path_str = split + "." + str(last_shard - 1) + ".json"
    if compression:
        chunk_file_path_str += ".gz"
    chunk_file_path = os.path.join(output_path, chunk_file_path_str)

    line_source = source_file.readline().strip()

    line_source_tokenized = next(tokenize(nlp, [line_source]))

    # Apply preprocessing on the line
    preprocessed_line = preprocess(
        line_source_tokenized,
        [1] * len(line_source_tokenized),
        args.min_sentence_ntokens,
        args.max_sentence_ntokens,
        args.min_example_nsents,
        args.max_example_nsents,
    )[0]

    try:
        chunk_json, _ = load_json(chunk_file_path)
    except FileNotFoundError:
        logger.error(
            "The file at path %s was not found. Make sure `--compression` is set correctly.",
            chunk_file_path,
        )
    last_item_chunk = chunk_json[-1]
    line_chunk = last_item_chunk["src"]

    # remove the last item if it is a newline
    if line_chunk[-1] == ["\n"]:
        line_chunk.pop()

    if line_chunk == preprocessed_line:
        logger.info("Resume Successful!")
        logger.debug("`source_file` moved forward one line")
    else:
        logger.info("Resume NOT Successful")
        logger.info("Last Chunk Line: %s", line_chunk)
        logger.info("Previous (to resume line) Source Line: %s", preprocessed_line)
        # skipcq: PYL-W1201
        logger.info(
            "Common causes of this issue:\n"
            + "1. You changed the `--shard_interval`. You used a different interval previously "
            + "than you used in the command to resume.\n"
            + "2. The abstractive (`.source` and `.target`) or extractive (`.json`) dataset "
            + "files were modified or removed. The last `.json` file needs to be in the same "
            + "folder it was originally outputted to so the last shard index and be determined "
            + "and the last line can be read.\n"
            + "3. It is entirely possible that there is a bug in this script. If you have checked "
            + "that the above were not the cause and that there were no issues pertaining to your "
            + "dataset then open an issue at https://github.com/HHousen/TransformerSum/issues/new."
        )
        return False

    return True


def seek_files(files, line_num):
    """Seek a set of files to line number ``line_num`` and return the files."""
    rtn_file_objects = []
    for file_object in files:
        offset = 0
        for idx, line in enumerate(file_object):
            if idx >= line_num:
                break
            offset += len(line)
        file_object.seek(0)

        file_object.seek(offset)
        rtn_file_objects.append(file_object)
    return rtn_file_objects


def save(json_to_save, output_path, compression=False):
    """
    Save ``json_to_save`` to ``output_path`` with optional gzip compresssion
    specified by ``compression``.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving to %s", output_path)
    if compression:
        # https://stackoverflow.com/a/39451012
        json_str = json.dumps(json_to_save)
        json_bytes = json_str.encode("utf-8")
        with gzip.open((output_path + ".gz"), "w") as save_file:
            save_file.write(json_bytes)
    else:
        with open(output_path, "w") as save_file:
            save_file.write(json.dumps(json_to_save))


def tokenize(
    nlp,
    docs,
    n_process=5,
    batch_size=100,
    name="",
    tokenizer_log_interval=0.1,
    disable_progress_bar=False,
):
    """Tokenize using spacy and split into sentences and tokens."""
    tokenized = []

    for doc in tqdm(
        nlp.pipe(
            docs,
            n_process=n_process,
            batch_size=batch_size,
        ),
        total=len(docs),
        desc="Tokenizing" + name,
        mininterval=tokenizer_log_interval,
        disable=disable_progress_bar,
    ):
        tokenized.append(doc)

    logger.debug("Splitting into sentences and tokens and converting to lists")
    t0 = time()

    doc_sents = (doc.sents for doc in tokenized)

    del tokenized
    del docs
    sents = (
        [[token.text for token in sentence] for sentence in doc] for doc in doc_sents
    )
    del doc_sents

    logger.debug("Done in %.2f seconds", time() - t0)
    # `sents` is an array of documents where each document is an array sentences where each
    # sentence is an array of tokens
    return sents


def example_processor(inputs, args, oracle_mode="greedy", no_preprocess=False):
    """
    Create ``oracle_ids``, convert them to ``labels`` and run
    :meth:`~convert_to_extractive.preprocess`.
    """
    source_doc, target_doc = inputs
    if oracle_mode == "greedy":
        oracle_ids = greedy_selection(source_doc, target_doc, 3)
    elif oracle_mode == "combination":
        oracle_ids = combination_selection(source_doc, target_doc, 3)

    # `oracle_ids` to labels
    labels = [0] * len(source_doc)
    for l_id in oracle_ids:
        labels[l_id] = 1

    # The number of sentences in the source document should equal the number of labels.
    # There should be one label per sentence.
    assert len(source_doc) == len(labels), (
        "Document: "
        + str(source_doc)
        + "\nLabels: "
        + str(labels)
        + "\n^^ The above document and label combination are not equal in length. The cause of "
        + "this problem in not known. This check exists to prevent further problems down the "
        + "data processing pipeline."
    )

    if no_preprocess:
        preprocessed_data = source_doc, labels
    else:
        preprocessed_data = preprocess(
            source_doc,
            labels,
            args.min_sentence_ntokens,
            args.max_sentence_ntokens,
            args.min_example_nsents,
            args.max_example_nsents,
        )

    return preprocessed_data, target_doc


def preprocess(
    example,
    labels,
    min_sentence_ntokens=5,
    max_sentence_ntokens=200,
    min_example_nsents=3,
    max_example_nsents=100,
):
    """
    Removes sentences that are too long/short and examples that have
    too few/many sentences.
    """
    # pick the sentence indexes in `example` if they are larger then `min_sentence_ntokens`
    idxs = [i for i, s in enumerate(example) if (len(s) > min_sentence_ntokens)]
    # truncate selected source sentences to `max_sentence_ntokens`
    example = [example[i][:max_sentence_ntokens] for i in idxs]
    # only pick labels for sentences that matched the length requirement
    labels = [labels[i] for i in idxs]
    # truncate entire source to max number of sentences (`max_example_nsents`)
    example = example[:max_example_nsents]
    # perform above truncation to `labels`
    labels = labels[:max_example_nsents]

    # if the example does not meet the length requirement then return None
    if len(example) < min_example_nsents:
        return None
    return example, labels


# Section Methods (to convert abstractive summary to extractive)
# Copied from https://github.com/nlpyang/BertSum/blob/9aa6ab84faf3a50724ce7112c780a4651de289b0/src/prepro/data_builder.py  # noqa: E501
def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations(
            [i for i in range(len(sents)) if i not in impossible_sents], s + 1
        )
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]

            rouge_score = rouge_1 + rouge_2
            if s == 0 and rouge_score == 0:
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for _ in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
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


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert an Abstractive Summarization Dataset to the Extractive Task"
    )

    parser.add_argument(
        "base_path", metavar="DIR", type=str, help="path to data directory"
    )
    parser.add_argument(
        "--base_output_path",
        type=str,
        default=None,
        help="path to output processed data (default is `base_path`)",
    )
    parser.add_argument(
        "--split_names",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        nargs="+",
        help="which splits of dataset to process",
    )
    parser.add_argument(
        "--add_target_to",
        default=["test"],
        choices=["train", "val", "test"],
        nargs="+",
        help="add the abstractive target to these splits (useful for calculating rouge scores)",
    )
    parser.add_argument(
        "--source_ext", type=str, default="source", help="extension of source files"
    )
    parser.add_argument(
        "--target_ext", type=str, default="target", help="extension of target files"
    )
    parser.add_argument(
        "--oracle_mode",
        type=str,
        default="greedy",
        choices=["greedy", "combination"],
        help="method to convert abstractive summaries to extractive summaries",
    )
    parser.add_argument(
        "--shard_interval",
        type=int,
        default=None,
        help="how many examples to include in each shard of the dataset (default: no shards)",
    )
    parser.add_argument(
        "--n_process",
        type=int,
        default=6,
        help="number of processes for multithreading",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="number of batches for tokenization"
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help="use gzip compression when saving data",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from last shard",
    )
    parser.add_argument(
        "--tokenizer_log_interval",
        type=float,
        default=0.1,
        help="minimum progress display update interval [default: 0.1] seconds",
    )
    parser.add_argument(
        "--sentencizer",
        action="store_true",
        help="use a spacy sentencizer instead of a statistical model for sentence "
        + "detection (much faster but less accurate); see https://spacy.io/api/sentencizer",
    )
    parser.add_argument(
        "--no_preprocess",
        action="store_true",
        help="do not run the preprocess function, which removes sentences that are too "
        + "long/short and examples that have too few/many sentences",
    )
    parser.add_argument(
        "--min_sentence_ntokens",
        type=int,
        default=5,
        help="minimum number of tokens per sentence",
    )
    parser.add_argument(
        "--max_sentence_ntokens",
        type=int,
        default=200,
        help="maximum number of tokens per sentence",
    )
    parser.add_argument(
        "--min_example_nsents",
        type=int,
        default=3,
        help="minimum number of sentences per example",
    )
    parser.add_argument(
        "--max_example_nsents",
        type=int,
        default=100,
        help="maximum number of sentences per example",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The dataset name from the `nlp` library to use for training/evaluation/testing. "
        + "Default is None.",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        default=None,
        help="The version of the dataset specified by `--dataset`. Default is None.",
    )
    parser.add_argument(
        "--data_example_column",
        type=str,
        default=None,
        help="The column of the `nlp` dataset that contains the text to be summarized. "
        + "Default is None.",
    )
    parser.add_argument(
        "--data_summarized_column",
        type=str,
        default=None,
        help="The column of the `nlp` dataset that contains the summarized text. Default is None.",
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )
    main_args = parser.parse_args()

    if main_args.resume and not main_args.shard_interval:
        parser.error(
            "Resuming requires both shard mode (--shard_interval) to be enabled and "
            + "shards to be created. Must use same 'shard_interval' that was used "
            + "previously to create the files to be resumed from."
        )

    # The `nlp` library has specific names for the dataset split names so set them
    # if using a dataset from `nlp`
    if main_args.dataset:
        main_args.split_names = ["train", "validation", "test"]

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(main_args.logLevel),
    )

    convert_to_extractive_driver(main_args)
