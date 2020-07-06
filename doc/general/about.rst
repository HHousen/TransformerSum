About
=====

Overview
--------

``TransformerSum`` is a library that aims to make to easy to *train*, *evaluate*, and *use* machine learning **transformer models** that perform **automatic summarization**. It features tight integration with `huggingface/transformers <https://github.com/huggingface/transformers>`_ which enables the easy usage of a **wide variety of architectures** and **pre-trained models**. There is a heavy emphasis on code **readability** and **interpretability** so that both beginners and experts can build new components. Both the extractive and abstractive model classes are written using `pytorch_lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_, which handles the PyTorch training loop logic, enabling **easy usage of advanced features** such as 16-bit precision, multi-GPU training, and `much more <https://pytorch-lightning.readthedocs.io/>`__. ``TransformerSum`` supports both the extractive and abstractive summarization of **long sequences** (4,096 to 16,000 tokens) using the `longformer <https://huggingface.co/transformers/model_doc/longformer.html>`__ (extractive) and `longbart <https://github.com/patil-suraj/longbart>`__ (abstractive), which is a combination of `BART <https://huggingface.co/transformers/model_doc/bart.html>`_ (`paper <https://arxiv.org/abs/1910.13461>`__) and the longformer. Models are automatically evaluated with the **ROUGE metric** but human tests can be conducted by the user.

Check out the :ref:`installation instructions <installation_instructions>` and the :ref:`tutorial <getting_started_tutorial>` to get started training models and summarizing text.

Both extractive and abstractive processed datasets and trained models can be found in their respective sections. Alternatively, you can browse from the root folder in Google Drive that contains all of the models and datasets: `TransformerSum Root Folder <https://drive.google.com/drive/folders/1SX8iQdUJkaLu8K6SoU0nrsxwOe4Qno6l>`_.

Features
--------

* For extractive summarization, compatible with every `huggingface/transformers <https://github.com/huggingface/transformers>`_ transformer encoder model.
* For abstractive summarization, compatible with every `huggingface/transformers <https://github.com/huggingface/transformers>`_ EncoderDecoder model.
* Currently, 10+ pre-trained extractive models available to summarize text trained on 3 datasets (CNN-DM, WikiHow, and ArXiv-PebMed).

* Contains pre-trained models that excel at summarization on resource-limited devices: On CNN-DM, ``mobilebert-uncased-ext-sum`` achieves about 97% of the performance of `BertSum <https://arxiv.org/abs/1903.10318>`_ while containing 4.45 times fewer parameters. It achieves about 94% of the performance of `MatchSum (Zhong et al., 2020) <https://arxiv.org/abs/2004.08795>`_, the current extractive state-of-the-art.
* Contains code to train models that excel at summarizing long sequences: The `longformer <https://huggingface.co/transformers/model_doc/longformer.html>`__ (extractive) and `longbart <https://github.com/patil-suraj/longbart>`__ (abstractive) can summarize sequences of lengths up to 4,096 tokens by default, but can be trained to summarize sequences of more than 16k.

* Integration with `huggingface/nlp <https://github.com/huggingface/nlp>`_ means any summarization dataset in the ``nlp`` library can be used for both abstractive and extractive training.
* "Smart batching" support to not perform unnecessary calculations (speeds up training).
* Use of ``pytorch_lighting`` for code readability.
* Extensive documentation.
* Two pooling modes (convert word vectors to sentence embeddings): mean of word embeddings or use the CLS token.

Significant People
------------------

The project was created by `Hayden Housen <https://haydenhousen.com/>`_ during his sophomore year of highschool as part of the Science Research program. It is actively maintained and updated by him and the community. You can contribute at `HHousen/TransformerSum <https://github.com/HHousen/TransformerSum>`_.

.. _about_rouge_scores:

ROUGE Scores
------------

This project uses ROUGE to evaluate summarization quality. ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics used to evaluate automatic summarization systems. However, automatic metrics, such as ROUGE and METEOR, have serious limitations. They only assess content selection by calculating lexical overlap and do not account for other quality aspects, such as fluency, grammaticality, or coherence. More information about the limitations of ROUGE in `sebastianruder/NLP-progress <https://github.com/sebastianruder/NLP-progress/blob/master/english/summarization.md>`_.

Links:

* `ROUGE Paper <https://www.aclweb.org/anthology/W04-1013/>`_ (`PDF <https://www.aclweb.org/anthology/W04-1013.pdf>`__)
* `ROUGE Score Wikipedia <https://en.wikipedia.org/wiki/ROUGE_(metric)>`_
* `Overview of How ROUGE Works <https://kavita-ganesan.com/what-is-rouge-and-how-it-works-for-evaluation-of-summaries/>`_ (`Archive <https://web.archive.org/web/20200624011354/https://kavita-ganesan.com/what-is-rouge-and-how-it-works-for-evaluation-of-summaries/>`__)

This project integrates with `rouge-score <https://pypi.org/project/rouge-score/>`__ and `pyrouge <https://pypi.org/project/pyrouge/>`__ and either can be used when calculating ROUGE scores during the testing phase.

``rouge-score`` is the default option. It is a pure python implementation of ROUGE designed to replicate the results of the official ROUGE package. While this option is cleaner (no perl installation required, no temporary directories, faster processing) than using ``pyrouge``, this option should not be used for official results due to minor score differences with ``pyrouge``.

``pyrouge`` is a python interface to the official ROUGE 1.5.5 perl script. Using this option will produce official scores, but it requires a complicated setup. To install ROUGE 1.5.5 I followed `this StackOverflow answer <https://stackoverflow.com/a/28941840>`_ and ran the below `commands from Kavita Ganesan <https://kavita-ganesan.com/rouge-howto/>`_ (`Archive <https://web.archive.org/web/20200624011208/https://kavita-ganesan.com/rouge-howto/>`__) to fix the WordNet exceptions:

.. code-block:: 

    cd data/WordNet-2.0-Exceptions/
    ./buildExeptionDB.pl . exc WordNet-2.0.exc.db

    cd ../
    rm WordNet-2.0.exc.db
    ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db

.. note:: The official ROUGE website was http://www.berouge.com/Pages/default.aspx but has been offline for many years. The Internet Archive still has a copy `here <https://web.archive.org/web/20160402021817/http://www.berouge.com/Pages/default.aspx>`__. However, you can still download ROUGE 1.5.5 from `andersjo/pyrouge <https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5>`_.

You can compute the ROUGE scores between a candidate text file and a ground-truth text file where each file contains one summary per line with the following command:

.. code-block:: 

    python -c "import helpers; helpers.test_rouge('tmp', 'save_gold.txt', 'save_pred.txt')"

Two flavors of ROUGE-L
^^^^^^^^^^^^^^^^^^^^^^

In the ROUGE paper, two flavors of ROUGE-L are described:

    sentence-level: Compute longest common subsequence (LCS) between two pieces of text. Newlines are ignored. This is called rougeL in this package.
    summary-level: Newlines in the text are interpreted as sentence boundaries, and the LCS is computed between each pair of reference and candidate sentences, and something called union-LCS is computed. This is called ``rougeLsum`` in the `rouge-score <https://github.com/google-research/google-research/tree/master/rouge>`_ package.
