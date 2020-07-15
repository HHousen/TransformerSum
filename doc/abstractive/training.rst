Training an Abstractive Summarization Model
===========================================

You can finetune/train abstractive summarization models such as `BART <https://huggingface.co/transformers/model_doc/bart.html>`__ and `T5 <https://huggingface.co/transformers/model_doc/t5.html>`__ with this script. You can also train models consisting of any encoder and decoder combination with an `EncoderDecoderModel <https://huggingface.co/transformers/model_doc/encoderdecoder.html>`_ by specifying the ``--decoder_model_name_or_path`` option (the ``--model_name_or_path`` argument specifies the encoder when using this configuration).

Alternatives:

* While you can use this script to load a pre-trained `BART <https://arxiv.org/abs/1910.13461>`__ or `T5 <https://arxiv.org/abs/1910.10683>`__ model and perform inference, it is recommended to use a `huggingface/transformers summarization pipeline <https://huggingface.co/transformers/main_classes/pipelines.html#summarizationpipeline>`_.
* To summarize PDF documents efficiently check out `HHousen/DocSum <https://github.com/HHousen/DocSum>`_.
* To summarize documents and strings of text using `PreSumm <https://arxiv.org/abs/1908.08345>`_ please visit `HHousen/DocSum <https://github.com/HHousen/DocSum>`_.
* You can also use the `summarization examples in huggingface/transformers <https://github.com/huggingface/transformers/tree/master/examples/seq2seq>`_, which are similar to this script, to train a model for seq2seq tasks. Most notably, the huggingface scripts don't integrate with ``nlp`` for easy and efficient dataset processing or make it easy to train EncoderDecoderModels.

.. _abstractive_command_example:

Example
-------

Example training command:

.. code-block::

    python main.py \
    --mode abstractive \
    --model_name_or_path bert-base-uncased \
    --decoder_model_name_or_path bert-base-uncased \
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

.. warning:: Abstractive long summarization is a work in progress. Please see `huggingface/transformers #4406 <https://github.com/huggingface/transformers/issues/4406>`_ for more info.

This script can perform abstractive summarization on long sequences using the ``longbart`` model (`GitHub repo <https://github.com/patil-suraj/longbart>`__). ``longbart`` is `BART <https://huggingface.co/transformers/model_doc/bart.html>`__ (`paper <https://arxiv.org/abs/1910.13461>`__) but with components from the `longformer <https://huggingface.co/transformers/model_doc/longformer.html>`_ (`paper <https://arxiv.org/abs/2004.05150>`__) that enable it to operate with long sequences.

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

GitHub issues that discuss ``longbart``:

1. `huggingface/transformers #4406 <https://github.com/huggingface/transformers/issues/4406>`_
2. `allenai/longformer #28 <https://github.com/allenai/longformer/issues/28>`_

.. _abstractive_script_help:

Script Help
-----------

Output of ``python main.py --mode abstractive --help`` (:ref:`generic options <main_script_generic_options>` removed):

.. code-block::

    usage: main.py [-h]
                    [--model_name_or_path MODEL_NAME_OR_PATH]
                    [--decoder_model_name_or_path DECODER_MODEL_NAME_OR_PATH]
                    [--batch_size BATCH_SIZE] [--val_batch_size VAL_BATCH_SIZE]
                    [--test_batch_size TEST_BATCH_SIZE]
                    [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                    [--adam_epsilon ADAM_EPSILON] [--warmup_steps WARMUP_STEPS]
                    [--use_scheduler USE_SCHEDULER] [--weight_decay WEIGHT_DECAY]
                    [--only_preprocess] [--dataset DATASET]
                    [--dataset_version DATASET_VERSION]
                    [--data_example_column DATA_EXAMPLE_COLUMN]
                    [--data_summarized_column DATA_SUMMARIZED_COLUMN]
                    [--cache_file_path CACHE_FILE_PATH] [--split_char SPLIT_CHAR]
                    [--use_percentage_of_data USE_PERCENTAGE_OF_DATA]
                    [--save_percentage SAVE_PERCENTAGE] [--save_hg_transformer]
                    [--test_use_pyrouge] [--sentencizer] [--gen_max_len GEN_MAX_LEN]
                    [--label_smoothing LABEL_SMOOTHING] [--sortish_sampler]

        optional arguments:
        -h, --help            show this help message and exit
        --model_name_or_path MODEL_NAME_OR_PATH
                                Path to pre-trained model or shortcut name. A list of
                                shortcut names can be found at https://huggingface.co/t
                                ransformers/pretrained_models.html. Community-uploaded
                                models are located at https://huggingface.co/models.
        --decoder_model_name_or_path DECODER_MODEL_NAME_OR_PATH
                                Path to pre-trained model or shortcut name to use as
                                the decoder. Default is the value of
                                `--model_name_or_path`.
        --batch_size BATCH_SIZE
                                Batch size per GPU/CPU for training/evaluation/testing.
        --val_batch_size VAL_BATCH_SIZE
                                Batch size per GPU/CPU for evaluation. This option
                                overwrites `--batch_size` for evaluation only.
        --test_batch_size TEST_BATCH_SIZE
                                Batch size per GPU/CPU for testing. This option
                                overwrites `--batch_size` for testing only.
        --dataloader_num_workers DATALOADER_NUM_WORKERS
                                The number of workers to use when loading data. A
                                general place to start is to set num_workers equal to
                                the number of CPUs on your machine. More details here:
                                https://pytorch-lightning.readthedocs.io/en/latest/perf
                                ormance.html#num-workers
        --adam_epsilon ADAM_EPSILON
                                Epsilon for Adam optimizer.
        --warmup_steps WARMUP_STEPS
                                Linear warmup over warmup_steps. Only active if
                                `--use_scheduler` is set.
        --use_scheduler USE_SCHEDULER
                                Two options: 1. `linear`: Use a linear schedule that
                                inceases linearly over `--warmup_steps` to
                                `--learning_rate` then decreases linearly for the rest
                                of the training process. 2. `onecycle`: Use the one
                                cycle policy with a maximum learning rate of
                                `--learning_rate`. (default: False, don't use any
                                scheduler)
        --weight_decay WEIGHT_DECAY
        --only_preprocess     Only preprocess and write the data to disk. Don't train
                                model.
        --dataset DATASET     The dataset name from the `nlp` library to use for
                                training/evaluation/testing. Default is
                                `cnn_dailymail`.
        --dataset_version DATASET_VERSION
                                The version of the dataset specified by `--dataset`.
        --data_example_column DATA_EXAMPLE_COLUMN
                                The column of the `nlp` dataset that contains the text
                                to be summarized. Default value is for the
                                `cnn_dailymail` dataset.
        --data_summarized_column DATA_SUMMARIZED_COLUMN
                                The column of the `nlp` dataset that contains the
                                summarized text. Default value is for the
                                `cnn_dailymail` dataset.
        --cache_file_path CACHE_FILE_PATH
                                Path to cache the tokenized dataset.
        --split_char SPLIT_CHAR
                                If the `--data_summarized_column` is already split into
                                sentences then use this option to specify which token
                                marks sentence boundaries. If the summaries are not
                                split into sentences then spacy will be used to split
                                them. The default is None, which means to use spacy.
        --use_percentage_of_data USE_PERCENTAGE_OF_DATA
                                When filtering the dataset, only save a percentage of
                                the data. This is useful for debugging when you don't
                                want to process the entire dataset.
        --save_percentage SAVE_PERCENTAGE
                                Percentage (divided by batch_size) between 0 and 1 of
                                the predicted and target summaries from the test set to
                                save to disk during testing. This depends on batch
                                size: one item from each batch is saved
                                `--save_percentage` percent of the time. Thus, you can
                                expect `len(dataset)*save_percentage/batch_size`
                                summaries to be saved.
        --save_hg_transformer
                                Save the `huggingface/transformers` model whenever a
                                checkpoint is saved.
        --test_use_pyrouge    Use `pyrouge`, which is an interface to the official
                                ROUGE software, instead of the pure-python
                                implementation provided by `rouge-score`. You must have
                                the real ROUGE package installed. More details about
                                ROUGE 1.5.5 here: https://github.com/andersjo/pyrouge/t
                                ree/master/tools/ROUGE-1.5.5. It is recommended to use
                                this option for official scores. The `ROUGE-L`
                                measurements from `pyrouge` are equivalent to the
                                `rougeLsum` measurements from the default `rouge-score`
                                package.
        --sentencizer         Use a spacy sentencizer instead of a statistical model
                                for sentence detection (much faster but less accurate)
                                during data preprocessing; see
                                https://spacy.io/api/sentencizer.
        --gen_max_len GEN_MAX_LEN
                                Maximum sequence length during generation while testing
                                and when using the `predict()` function.
        --label_smoothing LABEL_SMOOTHING
                                `LabelSmoothingLoss` implementation from OpenNMT
                                (https://bit.ly/2ObgVPP) as stated in the original
                                paper https://arxiv.org/abs/1512.00567.
        --sortish_sampler     Reorganize the input_ids by length with a bit of
                                randomness. This can help to avoid memory errors caused
                                by large batches by forcing large batches to be
                                processed first.