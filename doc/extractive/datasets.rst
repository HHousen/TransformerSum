.. _extractive_supported_datasets:

Extractive Supported Datasets
=============================

.. note:: In addition to the below datasets, all of the :ref:`abstractive datasets <abstractive_supported_datasets>` can be converted for extractive summarization and thus be used to train models. See :ref:`convert_to_extractive_option_2` for more information.

There are several ways to obtain and process the datasets below:

1. Download the converted extractive version for use with the training script (which will preprocess the data automatically (tokenization, etc.)). Note that all the provided extractive versions are split every 500 documents and are compressed. You will have to manually process if you desire different chunk sizes.
2. Download the processed abstractive version. This is the original data after being run through its respective processor located in the ``datasets`` folder.
3. Download the original data in its original form, which depends on how it was obtained in the original paper.

The table under each heading contains quick links to download the data. Beneath that are instructions to process the data manually.

.. _extractive_dataset_cnn_dm:

CNN/DM
------

The **CNN/DailyMail** (Hermann et al., 2015) dataset contains 93k articles from the CNN, and 220k articles the Daily Mail newspapers. Both publishers supplement their articles with bullet point summaries. Non-anonymized variant in See et al. (2017).

+-------------------------------+--------------------------------------------------------------------------------------+
| Type                          | Link                                                                                 |
+===============================+======================================================================================+
| Processor Repository          | `artmatsak/cnn-dailymail <https://github.com/artmatsak/cnn-dailymail>`_              |
+-------------------------------+--------------------------------------------------------------------------------------+
| Data Download Link            | `CNN/DM official website <https://cs.nyu.edu/~kcho/DMQA/>`__                         |
+-------------------------------+--------------------------------------------------------------------------------------+
| Processed Abstractive Dataset | `Google Drive <https://drive.google.com/uc?id=1jlo6kFUBxKZmTe4JHbGsdi5jPq-03-Ke>`__  |
+-------------------------------+--------------------------------------------------------------------------------------+
| Extractive Version            | `Google Drive <https://drive.google.com/uc?id=1ucg-6WpFm3hV_OGsE35Jq-UQrL3s4AJY>`__  |
+-------------------------------+--------------------------------------------------------------------------------------+

Download and unzip the stories directories from `here <https://cs.nyu.edu/~kcho/DMQA/>`_ for both CNN and Daily Mail. The files can be downloaded from the terminal with `gdown`, which can be installed with `pip install gdown`.

.. code-block:: bash

    pip install gdown
    gdown https://drive.google.com/uc?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ
    gdown https://drive.google.com/uc?id=0BwmD_VLjROrfM1BxdkxVaTY2bWs
    tar zxf cnn_stories.tgz
    tar zxf dailymail_stories.tgz

.. note:: The above Google Drive links may be outdated depending on the time you are reading this. Check the `CNN/DM official website <https://cs.nyu.edu/~kcho/DMQA/>`__ for the most up-to-date download links.

Next, run the processing code in the git submodule for `artmatsak/cnn-dailymail <https://github.com/artmatsak/cnn-dailymail>`_ located in ``datasets/cnn_dailymail_processor``. Run ``python make_datafiles.py /path/to/cnn/stories /path/to/dailymail/stories``, replacing `/`path/to/cnn/stories`` with the path to where you saved the ``cnn/stories`` directory that you downloaded; similarly for ``dailymail/stories``.

For each of the URL lists (``all_train.txt``, ``all_val.txt`` and ``all_test.txt``) in ``cnn_dailymail_processor/url_lists``, the corresponding stories are read from file and written to text files ``train.source``, ``train.target``, ``val.source``, ``val.target``, and ``test.source`` and ``test.target``. These will be placed in the newly created ``cnn_dm`` directory.

The original processing code is available at `abisee/cnn-dailymail <https://github.com/abisee/cnn-dailymail>`_, but for this project the `artmatsak/cnn-dailymail <https://github.com/artmatsak/cnn-dailymail>`_ processing code is used since it does not tokenize and writes the data to text file ``train.source``, ``train.target``, ``val.source``, ``val.target``, ``test.source`` and ``test.target``, which is the format expected by ``convert_to_extractive.py``.

WikiHow
-------

**WikiHow** (Koupaee and Wang, 2018) is a large-scale dataset of instructions from the online WikiHow.com website. Each of 200k examples consists of multiple instruction-step paragraphs along with a summarizing sentence. The task is to generate the concatenated summary-sentences from the paragraphs.

+------------------------+---------+
| Dataset Size           | 230,843 |
+========================+=========+
| Average Article Length | 579.8   |
+------------------------+---------+
| Average Summary Length | 62.1    |
+------------------------+---------+
| Vocabulary Size        | 556,461 |
+------------------------+---------+

+-------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Type                          | Link                                                                                                                                                                      |
+===============================+===========================================================================================================================================================================+
| Processor Repository          | `HHousen/WikiHow-Dataset <https://github.com/HHousen/WikiHow-Dataset>`_ (`Original Repo <https://github.com/mahnazkoupaee/WikiHow-Dataset>`__)                            |
+-------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Data Download Link            | `wikihowAll.csv <https://bit.ly/3cueodA>`_ (`mirror <https://drive.google.com/uc?id=1cl11A9aDcWxGn8qMibEy7ovY9UnOVzP6>`_) and `wikihowSep.csv <https://bit.ly/3btJ12G>`_  |
+-------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Processed Abstractive Dataset | `Google Drive <https://drive.google.com/uc?id=1v5rDxh5WrXYh-u8eyHfNKTKy89QI7Dkd>`__                                                                                       |
+-------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Extractive Version            | `Google Drive <https://drive.google.com/uc?id=1jyNdc0fhrriZXArV-9R2UJMHfdQaQV3D>`__                                                                                       |
+-------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Processing Steps:

1. Download `wikihowAll.csv <https://bit.ly/3cueodA>`_ (`main repo <https://github.com/mahnazkoupaee/WikiHow-Dataset>`__ for most up-to-date links) to ``datasets/wikihow_processor``
2. Run ``python process.py`` (runtime: 2m), which will create a new directory called ``wikihow`` containing the ``train.source``, ``train.target``, ``val.source``, ``val.target``, ``test.source`` and ``test.target`` files necessary for `convert_to_extractive.py`.

PubMed/ArXiv
------------

**ArXiv and PubMed** (Cohan et al., 2018) are two long document datasets of scientific publications
from [arXiv.org](http://arxiv.org/) (113k) and PubMed (215k). The task is to generate the abstract from the paper body.

+-----------------------+--------+--------------------------+-----------------------------+
| Datasets              | # docs | avg. doc. length (words) | avg. summary length (words) |
+=======================+========+==========================+=============================+
| CNN                   | 92K    | 656                      | 43                          |
+-----------------------+--------+--------------------------+-----------------------------+
| Daily Mail            | 219K   | 693                      | 52                          |
+-----------------------+--------+--------------------------+-----------------------------+
| NY Times              | 655K   | 530                      | 38                          |
+-----------------------+--------+--------------------------+-----------------------------+
| PubMed (this dataset) | 133K   | 3016                     | 203                         |
+-----------------------+--------+--------------------------+-----------------------------+
| arXiv (this dataset)  | 215K   | 4938                     | 220                         |
+-----------------------+--------+--------------------------+-----------------------------+

+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| Type                          | Link                                                                                                                                                   |
+===============================+========================================================================================================================================================+
| Processor Repository          | `HHousen/ArXiv-PubMed-Sum <https://github.com/HHousen/ArXiv-PubMed-Sum>`_ (`Original Repo <https://github.com/armancohan/long-summarization>`__)       |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| Data Download Link            | `PubMed <https://bit.ly/2VsKNvt>`_ (`mirror <https://bit.ly/2VLPJuh>`__) and `ArXiv <https://bit.ly/2wWeVpp>`_ (`mirror <https://bit.ly/2VPWnzs>`__)   |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| Processed Abstractive Dataset | `Google Drive <https://drive.google.com/uc?id=1a32oPIzIJ7DGekKL1tyAgfEyoI9NxENs>`__                                                                    |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+
| Extractive Version            | `Google Drive <https://drive.google.com/uc?id=1SsenAqbK1wmvd_1oWgAT1fRtLcA9rglS>`__                                                                    |
+-------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------+

Processing Steps:

1. Download `PubMed <https://bit.ly/2VsKNvt>`_ and `ArXiv <https://bit.ly/2wWeVpp>`_ (`main repo <https://github.com/armancohan/long-summarization>`__ for most up-to-date links) to ``datasets/arxiv-pubmed_processor``
2. Run the command ``python process.py <arxiv_articles_dir> <pubmed_articles_dir>`` (runtime: 5-10m), which will create a new directory called ``arxiv-pubmed`` containing the ``train.source``, ``train.target``, ``val.source``, ``val.target``, ``test.source`` and ``test.target`` files necessary for `convert_to_extractive.py`.

See the `repository's README.md <https://github.com/HHousen/ArXiv-PubMed-Sum/blob/master/README.md>`_.

.. note:: To convert this dataset to extractive it is recommended to use the ``--sentencizer`` option due to the size of the dataset. Additionally, the ``--max_sentence_ntokens`` should be set to ``300`` and the ``--max_example_nsents`` should be set to ``600``. See the :ref:`convert_to_extractive` section for more information. The full command should be similar to:

.. code-block:: bash

    python convert_to_extractive.py ./datasets/arxiv-pubmed_processor/arxiv-pubmed \
    --shard_interval 5000 \
    --sentencizer \
    --max_sentence_ntokens 300 \
    --max_example_nsents 600
