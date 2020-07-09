.. _abstractive_supported_datasets:

Abstractive Supported Datasets
==============================

All of the summarization datasets from the `huggingface/nlp <https://github.com/huggingface/nlp>`_ library are supported. Only 4 options (specifically ``--dataset``, ``--dataset_version``, ``--data_example_column``, and ``--data_summarized_column``) have to be changed to train a model on a new dataset.

The most notable datasets (the ones pertaining to summarization) are listed below.

.. warning:: The ``nlp`` library uses arrow files which are not compressed and can become large quite quickly. Thus, depending on your internet connection, hardware, and the size of the dataset it might be faster to reprocess the data than to download the pre-processed data.

If you download the preprocessed data, you can use it by setting the ``--cache_file_path`` option to the path containing the ``train_tokenized``, ``validation_tokenized``, and ``test_tokenized`` files.

+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| Dataset Name          | Processing Time | Preprocessed Data Download                                                                       |
+=======================+=================+==================================================================================================+
| ``cnn_dailymail``     | 30m / 59m       | `bert-base-uncased <https://bit.ly/38fMUHT>`__ & `longbart-base-4096 <https://bit.ly/3i5TCEJ>`__ |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``scientific_papers`` | 4h-5h           | `longbart-base-4096 <https://bit.ly/2O93r6S>`__                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``newsroom``          |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``reddit``            |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``multi_news``        |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``gigaword``          |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``billsum``           |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``wikihow``           |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``redit_tifu``        |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``xsum``              |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+
| ``opinosis``          |                 |                                                                                                  |
+-----------------------+-----------------+--------------------------------------------------------------------------------------------------+

The `live nlp viewer <https://huggingface.co/nlp/viewer>`_ visualizes the data and describes each dataset.