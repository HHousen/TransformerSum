Training an Abstractive Summarization Model
===========================================

.. _abstractive_command_example:

Example
-------

Example training command:

.. code-block::

    python main.py \
    --mode abstractive \
    --model_name_or_path bert-base-uncased \
    --cache_file_path data \
    --max_epochs 4 \
    --do_train --do_test \
    --batch_size 4 \
    --weights_save_path model_weights \
    --no_wandb_logger_log_model \
    --accumulate_grad_batches 5 \
    --use_scheduler linear \
    --warmup_steps 8000 \
    --gradient_clip_val 1.0 \
    --custom_checkpoint_every_n 300

This command will train and test a bert-to-bert model for abstractive summarization for 4 epochs with a batch size of 4. The weights are saved to ``model_weights/`` and will not be uploaded to wandb.ai due to the ``--no_wandb_logger_log_model`` option. The CNN/DM dataset (which is the default dataset) will be downloaded (and automatically processed) to ``data/``\ . The gradients will be accumulated every 5 batches and training will be optimized by AdamW with a scheduler that warms up linearly for 8000 then decays. A checkpoint file will be saved every 300 steps.

.. _abstractive_long_summarization:

Abstractive Long Summarization
------------------------------

This script can perform abstractive summarization on long sequences using the ``longbart`` model (`GitHub repo <https://github.com/patil-suraj/longbart>`__). ``longbart`` is `BART <https://huggingface.co/transformers/model_doc/bart.html>`_ (`paper <https://arxiv.org/abs/1910.13461>`__) but with components from the `longformer <https://huggingface.co/transformers/model_doc/longformer.html>`_ (`paper <https://arxiv.org/abs/2004.05150>`__) that enable it to operate with long sequences.

Install ``longbart`` by running ``pip install git+https://github.com/patil-suraj/longbart.git``. Then generate a long model with the below code:

.. code-block:: python

    import os
    from longbart.convert_bart_to_longbart import create_long_model

    model_path = 'longbart-base-4096'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model, tokenizer = create_long_model(
        save_model_to=model_path,
        base_model='facebook/bart-base',
        tokenizer_name_or_path='facebook/bart-base',
        attention_window=512,
        max_pos=4096
    )

You can change ``max_pos`` to a different value to summarize sequences longer than 4096 tokens.

With the ``longbart`` model generated you can run the training script with the ``--model_name_or_path`` set to ``longbart-base-4096`` (or wherever the configuration and model files are located).

.. warning:: For this script to work correctly with ``longbart`` the ``--model_name_or_path`` must contain the phrase "longbart".

.. _abstractive_script_help:

Script Help
-----------

Output of ``python main.py --mode abstractive --help`` (:ref:`generic options <main_script_generic_options>` removed):

.. code-block::

    usage: main.py [-h]
                    [--model_name_or_path MODEL_NAME_OR_PATH] [--batch_size BATCH_SIZE]
                    [--val_batch_size VAL_BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE]
                    [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                    [--adam_epsilon ADAM_EPSILON] [--warmup_steps WARMUP_STEPS]
                    [--use_scheduler USE_SCHEDULER] [--weight_decay WEIGHT_DECAY]
                    [--dataset DATASET] [--dataset_version DATASET_VERSION]
                    [--data_example_column DATA_EXAMPLE_COLUMN]
                    [--data_summarized_column DATA_SUMMARIZED_COLUMN]
                    [--save_percentage SAVE_PERCENTAGE] [--save_hg_transformer]

        optional arguments:
        -h, --help            show this help message and exit
        --model_name_or_path MODEL_NAME_OR_PATH
                                Path to pre-trained model or shortcut name. A list of shortcut
                                names can be found at
                                https://huggingface.co/transformers/pretrained_models.html.
                                Community-uploaded models are located at
                                https://huggingface.co/models.
        --batch_size BATCH_SIZE
                                Batch size per GPU/CPU for training/evaluation/testing.
        --val_batch_size VAL_BATCH_SIZE
                                Batch size per GPU/CPU for evaluation. This option overwrites
                                `--batch_size` for evaluation only.
        --test_batch_size TEST_BATCH_SIZE
                                Batch size per GPU/CPU for testing. This option overwrites
                                `--batch_size` for testing only.
        --dataloader_num_workers DATALOADER_NUM_WORKERS
                                The number of workers to use when loading data. A general place
                                to start is to set num_workers equal to the number of CPUs on
                                your machine. More details here: https://pytorch-
                                lightning.readthedocs.io/en/latest/performance.html#num-workers
        --adam_epsilon ADAM_EPSILON
                                Epsilon for Adam optimizer.
        --warmup_steps WARMUP_STEPS
                                Linear warmup over warmup_steps. Only active if `--use_scheduler`
                                is set.
        --use_scheduler USE_SCHEDULER
                                Two options: 1. `linear`: Use a linear schedule that inceases
                                linearly over `--warmup_steps` to `--learning_rate` then
                                decreases linearly for the rest of the training process. 2.
                                `onecycle`: Use the one cycle policy with a maximum learning rate
                                of `--learning_rate`. (default: False, don't use any scheduler)
        --weight_decay WEIGHT_DECAY
        --dataset DATASET     The dataset name from the `nlp` library to use for
                                training/evaluation/testing. Default is `cnn_dailymail`.
        --dataset_version DATASET_VERSION
                                The version of the dataset specified by `--dataset`.
        --data_example_column DATA_EXAMPLE_COLUMN
                                The column of the `nlp` dataset that contains the text to be
                                summarized. Default value is for the `cnn_dailymail` dataset.
        --data_summarized_column DATA_SUMMARIZED_COLUMN
                                The column of the `nlp` dataset that contains the summarized
                                text. Default value is for the `cnn_dailymail` dataset.
        --save_percentage SAVE_PERCENTAGE
                                Percentage (divided by batch_size) between 0 and 1 of the
                                predicted and target summaries from the test set to save to disk
                                during testing. This depends on batch size: one item from each
                                batch is saved `--save_percentage` percent of the time. Thus, you
                                can expect `len(dataset)*save_percentage/batch_size` summaries to
                                be saved.
        --save_hg_transformer
                                Save the `huggingface/transformers` model whenever a checkpoint
                                is saved.