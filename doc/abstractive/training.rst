Training an Abstractive Summarization Model
===========================================

You can finetune/train abstractive summarization models such as `BART <https://huggingface.co/transformers/model_doc/bart.html>`__ and `T5 <https://huggingface.co/transformers/model_doc/t5.html>`__ with this script. You can also train models consisting of any encoder and decoder combination with an `EncoderDecoderModel <https://huggingface.co/transformers/model_doc/encoderdecoder.html>`_ by specifying the ``--decoder_model_name_or_path`` option (the ``--model_name_or_path`` argument specifies the encoder when using this configuration).

Alternatives:

* While you can use this script to load a pre-trained `BART <https://arxiv.org/abs/1910.13461>`__ or `T5 <https://arxiv.org/abs/1910.10683>`__ model and perform inference, it is recommended to use a `huggingface/transformers summarization pipeline <https://huggingface.co/transformers/main_classes/pipelines.html#summarizationpipeline>`_.
* To summarize PDF documents efficiently check out `HHousen/DocSum <https://github.com/HHousen/DocSum>`_.
* To summarize documents and strings of text using `PreSumm <https://arxiv.org/abs/1908.08345>`_ please visit `HHousen/DocSum <https://github.com/HHousen/DocSum>`_.
* You can also use the `summarization examples in huggingface/transformers <https://github.com/huggingface/transformers/tree/master/examples/seq2seq>`_, which are similar to this script, to train a model for seq2seq tasks. Most notably, the huggingface scripts don't integrate with ``nlp`` for easy and efficient dataset processing or make it easy to train EncoderDecoderModels.

.. note:: Version 3.1.0 of huggingface/transformers enhances the encoder-decoder framework to allow for more encoder decoder model combinations such as Bert2GPT2, Roberta2Roberta, and Longformer2Roberta. Patrick von Platen has trained and tested some of these model combinations using custom scripts. His results can be found at `this huggingface/models page <https://huggingface.co/models?search=cnn_dailymail-fp16>`_.

The effectiveness of initializing sequence-to-sequence models with pre-trained checkpoints for sequence generation tasks was shown in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`_ by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Results for EncoderDecoderModels can be found in this paper. This script should be able to produce similar results to this paper.

.. important:: This script acts like a data preparation, training loop logic, and evaluation wrapper around models from ``huggingface/transformers``. For this reason, you can specify the ``--save_hg_transformer`` option, which will save the ``huggingface/transformers`` model whenever a checkpoint is saved using ``model.save_pretrained(save_path)``. Then, the trained model can be loaded without the ``TransformerSum`` library using just  ``huggingface/transformers`` in the future by running ``AutoModelForSeq2SeqLM.from_pretrained()`` (or ``EncoderDecoderModel.from_pretrained()`` if ``--decoder_model_name_or_path`` was used during training).


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

Importantly, you can specify the ``--tie_encoder_decoder`` option to tie the weights of the encoder and decoder when using an ``EncoderDecoderModel`` architecture. Specifying this option is equivalent to the "share" architecture tested in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`_.

.. _abstractive_long_summarization:

Abstractive Long Summarization
------------------------------

This script can perform abstractive summarization on long sequences using the `LongformerEncoderDecoder model <https://huggingface.co/transformers/model_doc/led.html>`_. ``LongformerEncoderDecoder`` is `BART <https://huggingface.co/transformers/model_doc/bart.html>`__ (`paper <https://arxiv.org/abs/1910.13461>`__) but with components from the `longformer <https://huggingface.co/transformers/model_doc/longformer.html>`_ (`paper <https://arxiv.org/abs/2004.05150>`__) that enable it to operate with long sequences.

During the development phase of the LED, LED installation and usage was complicated. Now, it is as simple as setting the ``--model_name_or_path`` option to a model from the `LED community models page <https://huggingface.co/models?filter=led>`__.

.. important:: You can adjust the sequence length that the trained LED model will be able to handle by modifying the ``--model_max_length`` argument. This option controls the length of sequences during the data processing stage. So, this option will have no effect with pre-compiled datasets listed under the "Preprocessed Data Download" heading on :ref:`abstractive_supported_datasets`.


GitHub issues that discussed the creation of ``LongformerEncoderDecoder``:

1. `huggingface/transformers #4406 <https://github.com/huggingface/transformers/issues/4406>`_
2. `allenai/longformer #28 <https://github.com/allenai/longformer/issues/28>`_

Step-by-Step Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Download dataset (≈2.8GB): ``gdown https://drive.google.com/uc?id=1rRzLwCl-s84Ji4ZfvfKV_iWunOy_cbE2``
2. Extract (≈90GB): ``tar -xzvf longformer-encdec-base-8192.tar.gz``
3. Training command:

    .. code-block::

        python main.py \
        --mode abstractive \
        --model_name_or_path allenai/led-base-16384 \
        --max_epochs 4 \
        --dataset scientific_papers \
        --do_train \
        --precision 16 \
        --amp_level O2 \
        --sortish_sampler \
        --batch_size 8 \
        --gradient_checkpointing \
        --label_smoothing 0.1 \
        --accumulate_grad_batches 2 \
        --use_scheduler linear \
        --warmup_steps 16000 \
        --gradient_clip_val 1.0 \
        --cache_file_path longformer-encdec-base-8192 \
        --nlp_cache_dir nlp-cache \
        --custom_checkpoint_every_n 18000

4. The ``--max_epochs``, ``--batch_size``, ``--accumulate_grad_batches``, ``--warmup_steps``, and ``--custom_checkpoint_every_n`` values will need to be tweaked.

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
                [--dataloader_num_workers DATALOADER_NUM_WORKERS] [--only_preprocess]
                [--no_prepare_data] [--dataset DATASET [DATASET ...]]
                [--dataset_version DATASET_VERSION] [--data_example_column DATA_EXAMPLE_COLUMN]
                [--data_summarized_column DATA_SUMMARIZED_COLUMN]
                [--cache_file_path CACHE_FILE_PATH] [--split_char SPLIT_CHAR]
                [--use_percentage_of_data USE_PERCENTAGE_OF_DATA]
                [--save_percentage SAVE_PERCENTAGE] [--save_hg_transformer] [--test_use_pyrouge]
                [--sentencizer] [--gen_max_len GEN_MAX_LEN] [--label_smoothing LABEL_SMOOTHING]
                [--sortish_sampler] [--nlp_cache_dir NLP_CACHE_DIR] [--tie_encoder_decoder]

    optional arguments:
    --model_name_or_path MODEL_NAME_OR_PATH
                            Path to pre-trained model or shortcut name. A list of shortcut names
                            can be found at
                            https://huggingface.co/transformers/pretrained_models.html. Community-
                            uploaded models are located at https://huggingface.co/models. Default
                            is 'bert-base-uncased'.
    --decoder_model_name_or_path DECODER_MODEL_NAME_OR_PATH
                            Path to pre-trained model or shortcut name to use as the decoder if an
                            EncoderDecoderModel architecture is desired. If this option is not
                            specified, the shortcut name specified by `--model_name_or_path` is
                            loaded using the Seq2seq AutoModel. Default is 'bert-base-uncased'.
    --batch_size BATCH_SIZE
                            Batch size per GPU/CPU for training/evaluation/testing.
    --val_batch_size VAL_BATCH_SIZE
                            Batch size per GPU/CPU for evaluation. This option overwrites
                            `--batch_size` for evaluation only.
    --test_batch_size TEST_BATCH_SIZE
                            Batch size per GPU/CPU for testing. This option overwrites
                            `--batch_size` for testing only.
    --dataloader_num_workers DATALOADER_NUM_WORKERS
                            The number of workers to use when loading data. A general place to
                            start is to set num_workers equal to the number of CPUs on your
                            machine. More details here: https://pytorch-
                            lightning.readthedocs.io/en/latest/performance.html#num-workers
    --only_preprocess     Only preprocess and write the data to disk. Don't train model.
    --no_prepare_data     Don't download, tokenize, or prepare data. Only load it from files.
    --dataset DATASET [DATASET ...]
                            The dataset name from the `nlp` library or a list of paths to Apache
                            Arrow files (that can be loaded with `nlp`) in the order train,
                            validation, test to use for training/evaluation/testing. Paths must
                            contain a '/' to be interpreted correctly. Default is `cnn_dailymail`.
    --dataset_version DATASET_VERSION
                            The version of the dataset specified by `--dataset`.
    --data_example_column DATA_EXAMPLE_COLUMN
                            The column of the `nlp` dataset that contains the text to be
                            summarized. Default value is for the `cnn_dailymail` dataset.
    --data_summarized_column DATA_SUMMARIZED_COLUMN
                            The column of the `nlp` dataset that contains the summarized text.
                            Default value is for the `cnn_dailymail` dataset.
    --cache_file_path CACHE_FILE_PATH
                            Path to cache the tokenized dataset.
    --split_char SPLIT_CHAR
                            If the `--data_summarized_column` is already split into sentences then
                            use this option to specify which token marks sentence boundaries. If
                            the summaries are not split into sentences then spacy will be used to
                            split them. The default is None, which means to use spacy.
    --use_percentage_of_data USE_PERCENTAGE_OF_DATA
                            When filtering the dataset, only save a percentage of the data. This is
                            useful for debugging when you don't want to process the entire dataset.
    --save_percentage SAVE_PERCENTAGE
                            Percentage (divided by batch_size) between 0 and 1 of the predicted and
                            target summaries from the test set to save to disk during testing. This
                            depends on batch size: one item from each batch is saved
                            `--save_percentage` percent of the time. Thus, you can expect
                            `len(dataset)*save_percentage/batch_size` summaries to be saved.
    --save_hg_transformer
                            Save the `huggingface/transformers` model whenever a checkpoint is
                            saved.
    --test_use_pyrouge    Use `pyrouge`, which is an interface to the official ROUGE software,
                            instead of the pure-python implementation provided by `rouge-score`.
                            You must have the real ROUGE package installed. More details about
                            ROUGE 1.5.5 here:
                            https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5. It
                            is recommended to use this option for official scores. The `ROUGE-L`
                            measurements from `pyrouge` are equivalent to the `rougeLsum`
                            measurements from the default `rouge-score` package.
    --sentencizer         Use a spacy sentencizer instead of a statistical model for sentence
                            detection (much faster but less accurate) during data preprocessing;
                            see https://spacy.io/api/sentencizer.
    --gen_max_len GEN_MAX_LEN
                            Maximum sequence length during generation while testing and when using
                            the `predict()` function.
    --label_smoothing LABEL_SMOOTHING
                            `LabelSmoothingLoss` implementation from OpenNMT
                            (https://bit.ly/2ObgVPP) as stated in the original paper
                            https://arxiv.org/abs/1512.00567.
    --sortish_sampler     Reorganize the input_ids by length with a bit of randomness. This can
                            help to avoid memory errors caused by large batches by forcing large
                            batches to be processed first.
    --nlp_cache_dir NLP_CACHE_DIR
                            Directory to cache datasets downloaded using `nlp`. Defaults to
                            '~/nlp'.
    --tie_encoder_decoder
                            Tie the encoder and decoder weights. Only takes effect when using an
                            EncoderDecoderModel architecture with the
                            `--decoder_model_name_or_path` option. Specifying this option is
                            equivalent to the 'share' architecture tested in 'Leveraging Pre-
                            trained Checkpoints for Sequence Generation Tasks'
                            (https://arxiv.org/abs/1907.12461).
