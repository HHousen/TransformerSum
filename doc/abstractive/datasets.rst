.. _abstractive_supported_datasets:

Abstractive Supported Datasets
==============================

All of the summarization datasets from the `huggingface/nlp <https://github.com/huggingface/nlp>`_ library are supported. Only 4 options (specifically ``--dataset``, ``--dataset_version``, ``--data_example_column``, and ``--data_summarized_column``) have to be changed to train a model on a new dataset.

The most notable datasets (the ones pertaining to summarization) are listed below.

.. warning:: The ``nlp`` library uses arrow files which are not heavily compressed and can become large quite quickly. Thus, depending on your internet connection, hardware, and the size of the dataset it might be faster to reprocess the data than to download the pre-processed data.

If you download the preprocessed data, you can use it by setting the ``--cache_file_path`` option to the path containing the ``train_tokenized``, ``validation_tokenized``, and ``test_tokenized`` files.

+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Dataset Name          | Processing Time | Preprocessed Data Download                                                                                                                                                                                      |
+=======================+=================+=================================================================================================================================================================================================================+
| ``cnn_dailymail``     | 30m / 59m       | `bert-base-uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/Abstractive/CNN-DM/cnn_dm_abs_preprocessed/bert-base-uncased/bert-base-uncased.tar.gz>`__ & `longformer-encdec-base-4096 <https://huggingface.co/HHousen/TransformerSum/blob/main/Abstractive/CNN-DM/cnn_dm_abs_preprocessed/longformer-encdec-base-4096/longformer-encdec-base-4096.tar.gz>`__   |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``scientific_papers`` | 2h-4h           | `longformer-encdec-base-4096 <https://huggingface.co/HHousen/TransformerSum/blob/main/Abstractive/arXiv-PubMed/arxiv_pubmed_abs_preprocessed/longformer-encdec-base-4096/longformer-encdec-base-4096.tar.gz>`__ & `longformer-encdec-base-8192 <https://huggingface.co/HHousen/TransformerSum/blob/main/Abstractive/arXiv-PubMed/arxiv_pubmed_abs_preprocessed/longformer-encdec-base-8192/longformer-encdec-base-8192.tar.gz>`__ |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``newsroom``          |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``reddit``            |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``multi_news``        |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``gigaword``          |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``billsum``           |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``wikihow``           |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``redit_tifu``        |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``xsum``              |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``opinosis``          |                 |                                                                                                                                                                                                                 |
+-----------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

The `live nlp viewer <https://huggingface.co/nlp/viewer>`_ visualizes the data and describes each dataset.

Custom Datasets
---------------

You can use a custom dataset with the abstractive training script. The data needs to be stored in three Apache Arrow files: training, validation, and testing. You can specify the article and summary columns with ``--data_example_column`` and ``--data_summarized_column``, respectfully. The ``--dataset`` argument allows multiple paths to be specified like so: ``--dataset /path/to/train /path/to/valid /path/to/test``. **The files must be specified in train, validation, test order and each path must have a slash ("/") in it.** Inputs without a slash are interpreted as ``huggingface/nlp`` dataset names. Essentially, if your data is stored in three Arrow files (one for each split) and has at least 2 columns with the article and summary, then it is supported.

A convince script is provided at ``scripts/convert_to_arrow.py`` to convert JSON datasets to the Arrow format. It will convert a list of JSON files to Arrow files and then combine them into one file. Each JSON file is sequentially loaded into RAM so only one JSON file will be in RAM at a time. The final combined file is memory mapped so RAM usage will be close to zero during the combination stage. The maximum memory usage throughout the duration of the script will be the size of the largest input JSON file.

Output of ``python scripts/convert_to_arrow.py --help``:

.. code-block::

    usage: convert_to_arrow.py [-h] [--file_paths FILE_PATHS [FILE_PATHS ...]] [--save_path SAVE_PATH]
                                [--cache_path_prefix CACHE_PATH_PREFIX] [--no_combine]

    optional arguments:
        -h, --help            show this help message and exit
        --file_paths FILE_PATHS [FILE_PATHS ...]
                                The paths to the JSON files to convert to arrow and combine.
        --save_path SAVE_PATH
                                The path to save the combined arrow file to. Defaults to './data.arrow'.
        --cache_path_prefix CACHE_PATH_PREFIX
                                The cache path and file name prefix for the converted JSON files. Defaults to './data_chunk'.
        --no_combine          Don't combine the converted JSON files.
