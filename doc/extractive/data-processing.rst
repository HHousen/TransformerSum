Automatic Preprocessing
=======================

While the ``convert_to_extractive.py`` script prepares a dataset for the extractive task, the data still needs to be processed for usage with a machine learning model. This preprocessing depends on the chosen model, and thus is implemented in the ``extractive.py`` file with the rest of the model training logic.

The actual :class:`~extractive.ExtractiveSummarizer` LightningModule (which is similar to an ``nn.Module`` but with a built-in training loop, more info at the `pytorch_lightning documentation <https://pytorch-lightning.readthedocs.io/en/latest/>`_) implements a :meth:`~extractive.ExtractiveSummarizer.prepare_data` function. This function is automatically called by ``pytorch_lightning`` to load and process the examples.

.. note:: Memory Usage Note: If sharding was turned off during the ``convert_to_extractive`` process then :meth:`~extractive.ExtractiveSummarizer.prepare_data` will run once, loading the entire dataset into memory to process just like the ``convert_to_extractive.py`` script.

There is a ``--only_preprocess`` argument available to only run the pre-process step and exit the script after all the examples have been written to disk. This option will force data to be preprocessed, even if it was already computed and is detected on disk, and any previous processed files will be overwritten.

Thus, to only pre-process data for use when training a model run:

.. code-block:: bash

    python main.py --data_path ./datasets/cnn_dailymail_processor/cnn_dm --use_logger tensorboard --model_name_or_path bert-base-uncased --model_type bert --do_train --only_preprocess

.. warning:: If processed files are detected, they will automatically be loaded from disk. This includes any files that follow the pattern ``[dataset_split_name].*.pt``, where ``*`` is any text of any length.
