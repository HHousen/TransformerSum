Training an Abstractive Summarization Model
===========================================

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