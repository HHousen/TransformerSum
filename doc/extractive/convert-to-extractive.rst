.. _convert_to_extractive:

Convert Abstractive to Extractive Dataset
=========================================

Overview
--------

Simply run `convert_to_extractive.py` with the path to the data. For example, with the :ref:`CNN/DM dataset <extractive_dataset_cnn_dm>`: ``python convert_to_extractive.py ./datasets/cnn_dailymail_processor/cnn_dm``. However, the recommended command is:

.. code-block:: bash

    python convert_to_extractive.py ./datasets/cnn_dailymail_processor/cnn_dm --shard_interval 5000 --compression --add_target_to test

* ``--shard_interval`` processes the file in chunks of ``5000`` and writes results to disk in chunks of ``5000`` (saves RAM)
* ``--compression`` compresses each output chunk with gzip (depending on the dataset reduces space usage requirement by about 1/2 to 1/3)
* ``--add_target_to`` will save the abstractive target text to the splits (in ``--split_names``) specified. 

The default output directory is the input directory that was specified, but the output directory can be changed with ``--base_output_path`` if desired.

If your files are not ``train``, ``val``, and ``test``, then the ``--split_names`` argument will let you specify the correct naming pattern. The ``--source_ext`` and ``--target_ext`` let you specify the file extension of the source and target files respectively. These must be different so the process can tell each section apart.

**Large Dataset? Need to Resume?:** The ``--resume`` option will read the output directory and determine on which document the script left off based on the shard_file names. If ``--shard_interval`` was ``None`` then resuming is not possible. Resuming is guaranteed to produce the same output as if ``--resume`` was not used because of :meth:`convert_to_extractive.check_resume_success()`, which checks to make sure the last line in the shard file is the same as the line directly before the line to resume with.

**Speed: Running Slowly?** There is a ``--sentencizer`` option to detect sentence boundaries without parsing dependencies. Instead of loading a statistical model using ``spacy``, this option will initialize the ``English`` `Language <https://spacy.io/api/language#init>`_ object and add a ``sentencizer`` to the `pipeline <https://spacy.io/api/language#create_pipe>`_. This is much faster than a `DependencyParser <https://spacy.io/api/dependencyparser>`_ but is also less accurate since the ``sentencizer`` uses a simpler, rule-based strategy.

Script Help
-----------

.. code-block::

    usage: convert_to_extractive.py [-h] [--base_output_path BASE_OUTPUT_PATH]
                                    [--split_names {train,val,test} [{train,val,test} ...]]
                                    [--add_target_to {train,val,test} [{train,val,test} ...]]
                                    [--source_ext SOURCE_EXT] [--target_ext TARGET_EXT]
                                    [--oracle_mode {greedy,combination}]
                                    [--shard_interval SHARD_INTERVAL]
                                    [--n_process N_PROCESS] [--batch_size BATCH_SIZE]
                                    [--compression] [--resume]
                                    [--tokenizer_log_interval TOKENIZER_LOG_INTERVAL]
                                    [--sentencizer] [--no_preprocess]
                                    [--min_sentence_ntokens MIN_SENTENCE_NTOKENS]
                                    [--max_sentence_ntokens MAX_SENTENCE_NTOKENS]
                                    [--min_example_nsents MIN_EXAMPLE_NSENTS]
                                    [--max_example_nsents MAX_EXAMPLE_NSENTS]
                                    [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                    DIR

    Convert an Abstractive Summarization Dataset to the Extractive Task

    positional arguments:
    DIR                   path to data directory

    optional arguments:
    -h, --help            show this help message and exit
    --base_output_path BASE_OUTPUT_PATH
                            path to output processed data (default is `base_path`)
    --split_names {train,val,test} [{train,val,test} ...]
                            which splits of dataset to process
    --add_target_to {train,val,test} [{train,val,test} ...]
                            add the abstractive target to these splits (useful for
                            calculating rouge scores)
    --source_ext SOURCE_EXT
                            extension of source files
    --target_ext TARGET_EXT
                            extension of target files
    --oracle_mode {greedy,combination}
                            method to convert abstractive summaries to extractive
                            summaries
    --shard_interval SHARD_INTERVAL
                            how many examples to include in each shard of the dataset
                            (default: no shards)
    --n_process N_PROCESS
                            number of processes for multithreading
    --batch_size BATCH_SIZE
                            number of batches for tokenization
    --compression         use gzip compression when saving data
    --resume              resume from last shard
    --tokenizer_log_interval TOKENIZER_LOG_INTERVAL
                            minimum progress display update interval [default: 0.1]
                            seconds
    --sentencizer         use a spacy sentencizer instead of a statistical model for
                            sentence detection (much faster but less accurate); see
                            https://spacy.io/api/sentencizer
    --no_preprocess       do not run the preprocess function, which removes sentences
                            that are too long/short and examples that have too few/many
                            sentences
    --min_sentence_ntokens MIN_SENTENCE_NTOKENS
                            minimum number of tokens per sentence
    --max_sentence_ntokens MAX_SENTENCE_NTOKENS
                            maximum number of tokens per sentence
    --min_example_nsents MIN_EXAMPLE_NSENTS
                            minimum number of sentences per example
    --max_example_nsents MAX_EXAMPLE_NSENTS
                            maximum number of sentences per example
    -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Set the logging level (default: 'Info').