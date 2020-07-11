import os
import time
import shutil
import json
import gzip
import logging
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def load_json(json_file):
    """Load a json file even if it is compressed with gzip.

    Args:
        json_file (str): Path to json file

    Returns:
        tuple: (documents, file_path), loaded json and path to file
    """
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
    return documents, file_path


class StepCheckpointCallback(pl.callbacks.base.Callback):
    def __init__(
        self, step_interval=1000, save_name="model", save_path=".", num_saves_to_keep=5
    ):
        super(StepCheckpointCallback, self).__init__()
        self.step_interval = step_interval
        self.save_name = save_name
        self.save_path = save_path
        self.num_saves_to_keep = num_saves_to_keep

    def on_batch_end(self, trainer, pl_module):
        # check if `step_interval` has passed and that the `global_step` is not 0
        if (
            trainer.global_step % self.step_interval == 0
            and not trainer.global_step == 0
        ):
            logger.info(
                "Saving model to "
                + str(self.save_path)
                + ".ckpt at step "
                + str(trainer.global_step)
                + "."
            )
            final_save_location = os.path.join(
                self.save_path,
                (self.save_name + "." + str(trainer.global_step) + ".ckpt"),
            )
            trainer.save_checkpoint(final_save_location)
            # remove previous saves
            offset = self.step_interval * self.num_saves_to_keep
            path_to_remove = (
                self.save_name + "." + str(trainer.global_step - offset) + ".ckpt"
            )
            if os.path.isfile(path_to_remove):
                os.remove(path_to_remove)


def lr_lambda_func(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def block_trigrams(candidate, prediction):
    """Decrease repetition in summaries by checking if a trigram from ``prediction`` 
    exists in ``candidate``

    Args:
        candidate (str): The string to check for trigrams from ``prediction``
        prediction (list): A list of strings to extract trigrams from

    Returns:
        bool: True if overlapping trigrams detected, False otherwise.
    """
    tri_c = _get_ngrams(3, candidate.split())
    for s in prediction:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False


def _get_ngrams(n, text):
    """Calculates n-grams.

    Args:
        n (int): which n-grams to calculate
        text (list): An array of tokens

    Returns:
        A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences."""
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def pad(data, pad_id, width=None, pad_on_left=False):
    """Pad ``data`` with ``pad_id`` to ``width`` on the right by default but if ``pad_on_left`` then left."""
    if not width:
        width = max(len(d) for d in data)
    if pad_on_left:
        rtn_data = [[pad_id] * (width - len(d)) + d for d in data]
    else:
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def pad_tensors(tensors, pad_id=0, width=None, pad_on_left=False):
    """Pad ``tensors`` with ``pad_id`` to ``width`` on the right by default but if ``pad_on_left`` then left."""
    if not width:
        width = max(len(d) for d in tensors)
    if pad_on_left:
        pad_params = ((width - len(tensor)), 0)
    else:
        pad_params = (0, (width - len(tensor)))
    return [
        F.pad(tensor, pad=pad_params, mode="constant", value=pad_id)
        for tensor in tensors
    ]


def test_rouge(temp_dir, cand, ref):
    """Compute ROUGE scores using the official ROUGE 1.5.5 package. This function uses the
    ``pyrouge`` python module to interface with the office ROUGE script. There should be a 
    "<q>" token between each sentence in the ``cand`` and ``ref`` files. ``pyrouge`` splits 
    sentences based on newlines but we cannot store all the summaries easily in a single text 
    file if there is a newline between each sentence since newlines mark new summaries. Thus, 
    the "<q>" token is used in the text files and is converted to a newline in this function.
    Using "<q>" instead of ``\\n`` also makes it easier to store the ground-truth summaries
    in the ``convert_to_extractive.py`` script.

    Args:
        temp_dir (str): A temporary folder to store files for input to the ROUGE script.
        cand (str): The path to the file containing one candidate summary per line with 
            "<q>" tokens in between each sentence.
        ref (str): The path to the file containing one ground-truth/gold summary per line 
            with "<q>" tokens in between each sentence.

    Returns:
        dict: Results from the ROUGE script as a python dictionary.
    """
    import pyrouge

    candidates = [line.strip() for line in open(cand, encoding="utf-8")]
    references = [line.strip() for line in open(ref, encoding="utf-8")]
    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(temp_dir, exist_ok=True)
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(tmp_dir + "/candidate", exist_ok=True)
    os.makedirs(tmp_dir + "/reference", exist_ok=True)

    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(
                tmp_dir + "/candidate/cand.{}.txt".format(i), "w", encoding="utf-8"
            ) as f:
                f.write(candidates[i].replace("<q>", "\n"))
            with open(
                tmp_dir + "/reference/ref.{}.txt".format(i), "w", encoding="utf-8"
            ) as f:
                f.write(references[i].replace("<q>", "\n"))
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = "ref.#ID#.txt"
        r.system_filename_pattern = r"cand.(\d+).txt"
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction="sum")

