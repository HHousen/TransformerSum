.. _train_extractive_model:

Training an Extractive Summarization Model
==========================================

Details
-------

Once the dataset has been converted to the extractive task, it can be used as input to a :class:`data.SentencesProcessor`, which has a :meth:`~data.SentencesProcessor.add_examples()` function to add sets of ``(example, labels)`` and a :meth:`~data.SentencesProcessor.get_features()` function that processes the data and prepares it to be inputted into the model (``input_ids``, ``attention_masks``, ``labels``, ``token_type_ids``, ``sent_rep_token_ids``, ``sent_rep_token_ids_masks``). Feature extraction runs in parallel and tokenizes text using the tokenizer appropriate for the model specified with ``--model_name_or_path``. The tokenizer can be changed to another ``huggingface/transformers`` tokenizer with the ``--tokenizer_name`` option.

.. important:: When loading a pre-trained model you may encounter this common error:

    .. code-block::

        RuntimeError: Error(s) in loading state_dict for ExtractiveSummarizer:
        Missing key(s) in state_dict: "word_embedding_model.embeddings.position_ids".

    To solve this issue, set ``strict=False`` like so: ``model = ExtractiveSummarizer.load_from_checkpoint("distilroberta-base-ext-sum.ckpt", strict=False)``. If you are using the ``main.py`` script, then you can alternatively sepcify the ``--no_strict`` option.

For the :ref:`CNN/DM dataset <extractive_dataset_cnn_dm>`, to train a model for 50,000 steps on the data run:

.. code-block:: bash

    python main.py --data_path ./datasets/cnn_dailymail_processor/cnn_dm --weights_save_path ./trained_models --do_train --max_steps 50000 --data_type txt

* The ``--do_train`` argument runs the training process. Set `--do_test` to test after training.
* The ``--data_path`` argument specifies where the extractive dataset json file are located.
* The `--weights_save_path` argument specifies where the model weights should be stored.

If you prefer to measure training progress by epochs instead of steps, you can use the ``--max_epochs`` and ``--min_epochs`` options.

The batch size can be changed with the ``--batch_size`` option. This changes the batch size for training, validation, and testing. You can set the ``--auto_scale_batch_size`` option to automatically determine this value. See `"Auto scaling of batch size" from the pytorch_lightning documentation <https://pytorch-lightning.readthedocs.io/en/0.7.6/training_tricks.html#auto-scaling-of-batch-size>`_ for more information about the algorithm and available options.

If the extractive dataset json files are compressed using gzip, then they will be automatically decompressed during the data preprocessing step of training.

By default, the model weights are saved after every epoch to the ``--default_root_dir``. The logs are also saved to this folder. You can change the weight save path (separate folder for logs and weights) with the ``--weights_save_path`` option.

The length of output summaries during testing is 3 by default. You can change this by setting the ``--test_k`` option to the number of sentences desired in generated summaries. This assumes ``--test_id_method`` is set to ``top_k``, which is the default. ``top_k`` selects the top ``k`` sentences and the other option, ``greater_k``, selects those sentences with a rank above ``k``. ``k`` is specified by the ``--test_k`` argument.

.. important:: More example training commands can be found on the `TransformerSum Weights & Biases page <https://app.wandb.ai/hhousen/transformerextsum>`__. Just click the name of a training run, go to the overview page by clicking the "i" icon in the top left, and look at the command value.

.. _extractive_pooling_modes:

Pooling Modes
-------------

The pooling model determines how word vectors should be converted to sentence embeddings. The implementation can be found in `pooling.py`. The ``--pooling_mode`` argument can be set to either ``sent_rep_tokens`` or ``mean_tokens``. While the pooling ``nn.Module`` allows multiple methods to be used at once (it will concatenate and return the results), the training script does not.

* ``sent_rep_tokens``: Uses the sentence representation token (commonly called the classification token; ``[CLS]`` in BERT and ``<s>`` in RoBERTa) vectors as sentence embeddings.
* ``mean_tokens``: Uses the average of the token vectors for each sentence in the input as sentence embeddings.
* ``max_tokens``: Uses the maximum of the token vectors for each sentence in the input as sentence embeddings.

Custom Models
-------------

You can use any `autoencoding transformer model <https://huggingface.co/transformers/model_summary.html#autoencoding-models>`_ for the word embedding model (by setting the ``--model_name_or_path`` CLI argument) as long as it was saved in the ``huggingface/transformers`` format. Any model that is loaded with this option by specifying a path is considered "custom" in this project. Currently, there are no "custom" models that are "officially" supported. The `longformer` used to be a custom model, but it was since added to the `huggingface/transformers` repository, and thus can be used in this project just like any other model.

.. _extractive_script_help:

Script Help
-----------

Output of ``python main.py --mode extractive --help`` (:ref:`generic options <main_script_generic_options>` removed):

.. code-block::

    usage: main.py [-h]
                [--model_name_or_path MODEL_NAME_OR_PATH] [--model_type MODEL_TYPE]
                [--tokenizer_name TOKENIZER_NAME] [--tokenizer_no_use_fast]
                [--max_seq_length MAX_SEQ_LENGTH] [--data_path DATA_PATH]
                [--data_type {txt,pt,none}] [--num_threads NUM_THREADS]
                [--processing_num_threads PROCESSING_NUM_THREADS]
                [--pooling_mode {sent_rep_tokens,mean_tokens,max_tokens}]
                [--num_frozen_steps NUM_FROZEN_STEPS] [--batch_size BATCH_SIZE]
                [--dataloader_type {map,iterable}]
                [--dataloader_num_workers DATALOADER_NUM_WORKERS]
                [--processor_no_bert_compatible_cls] [--only_preprocess]
                [--preprocess_resume] [--create_token_type_ids {binary,sequential}]
                [--no_use_token_type_ids]
                [--classifier {linear,simple_linear,transformer,transformer_linear}]
                [--classifier_dropout CLASSIFIER_DROPOUT]
                [--classifier_transformer_num_layers CLASSIFIER_TRANSFORMER_NUM_LAYERS]
                [--train_name TRAIN_NAME] [--val_name VAL_NAME]
                [--test_name TEST_NAME] [--test_id_method {greater_k,top_k}]
                [--test_k TEST_K] [--no_test_block_trigrams] [--test_use_pyrouge]
                [--loss_key {loss_total,loss_total_norm_batch,loss_avg_seq_sum,loss_avg_seq_mean,loss_avg}]

    optional arguments:
    -h, --help            show this help message and exit
    --model_name_or_path MODEL_NAME_OR_PATH
                            Path to pre-trained model or shortcut name. A list of
                            shortcut names can be found at https://huggingface.co/tran
                            sformers/pretrained_models.html. Community-uploaded models
                            are located at https://huggingface.co/models.
    --model_type MODEL_TYPE
                            Model type selected in the list: retribert, t5,
                            distilbert, albert, camembert, xlm-roberta, bart,
                            longformer, roberta, bert, openai-gpt, gpt2, mobilebert,
                            transfo-xl, xlnet, flaubert, xlm, ctrl, electra, reformer
    --tokenizer_name TOKENIZER_NAME
    --tokenizer_no_use_fast
                            Don't use the fast version of the tokenizer for the
                            specified model. More info: https://huggingface.co/transfo
                            rmers/main_classes/tokenizer.html.
    --max_seq_length MAX_SEQ_LENGTH
                            The maximum sequence length of processed documents.
    --data_path DATA_PATH
                            Directory containing the dataset.
    --data_type {txt,pt,none}
                            The file extension of the prepared data. The 'map'
                            `--dataloader_type` requires `txt` and the 'iterable'
                            `--dataloader_type` works with both. If the data is not
                            prepared yet (in JSON format) this value specifies the
                            output format after processing. If the data is prepared,
                            this value specifies the format to load. If it is `none`
                            then the type of data to be loaded will be inferred from
                            the `data_path`. If data needs to be prepared, this cannot
                            be set to `none`.
    --num_threads NUM_THREADS
    --processing_num_threads PROCESSING_NUM_THREADS
    --pooling_mode {sent_rep_tokens,mean_tokens,max_tokens}
                            How word vectors should be converted to sentence
                            embeddings.
    --num_frozen_steps NUM_FROZEN_STEPS
                            Freeze (don't train) the word embedding model for this
                            many steps.
    --batch_size BATCH_SIZE
                            Batch size per GPU/CPU for training/evaluation/testing.
    --dataloader_type {map,iterable}
                            The style of dataloader to use. `map` is faster and uses
                            less memory.
    --dataloader_num_workers DATALOADER_NUM_WORKERS
                            The number of workers to use when loading data. A general
                            place to start is to set num_workers equal to the number
                            of CPU cores on your machine. If `--dataloader_type` is
                            'iterable' then this setting has no effect and num_workers
                            will be 1. More details here: https://pytorch-
                            lightning.readthedocs.io/en/latest/performance.html#num-
                            workers
    --processor_no_bert_compatible_cls
                            If model uses bert compatible [CLS] tokens for sentence
                            representations.
    --only_preprocess     Only preprocess and write the data to disk. Don't train
                            model. This will force data to be preprocessed, even if it
                            was already computed and is detected on disk, and any
                            previous processed files will be overwritten.
    --preprocess_resume   Resume preprocessing. `--only_preprocess` must be set in
                            order to resume. Determines which files to process by
                            finding the shards that do not have a coresponding ".pt"
                            file in the data directory.
    --create_token_type_ids {binary,sequential}
                            Create token type ids during preprocessing.
    --no_use_token_type_ids
                            Set to not train with `token_type_ids` (don't pass them
                            into the model).
    --classifier {linear,simple_linear,transformer,transformer_linear}
                            Which classifier/encoder to use to reduce the hidden
                            dimension of the sentence vectors. `linear` - a
                            `LinearClassifier` with two linear layers, dropout, and an
                            activation function. `simple_linear` - a
                            `LinearClassifier` with one linear layer and a sigmoid.
                            `transformer` - a `TransformerEncoderClassifier` which
                            runs the sentence vectors through some
                            `nn.TransformerEncoderLayer`s and then a simple
                            `nn.Linear` layer. `transformer_linear` - a
                            `TransformerEncoderClassifier` with a `LinearClassifier`
                            as the `reduction` parameter, which results in the same
                            thing as the `transformer` option but with a
                            `LinearClassifier` instead of a `nn.Linear` layer.
    --classifier_dropout CLASSIFIER_DROPOUT
                            The value for the dropout layers in the classifier.
    --classifier_transformer_num_layers CLASSIFIER_TRANSFORMER_NUM_LAYERS
                            The number of layers for the `transformer` classifier.
                            Only has an effect if `--classifier` contains
                            "transformer".
    --train_name TRAIN_NAME
                            name for set of training files on disk (for loading and
                            saving)
    --val_name VAL_NAME   name for set of validation files on disk (for loading and
                            saving)
    --test_name TEST_NAME
                            name for set of testing files on disk (for loading and
                            saving)
    --test_id_method {greater_k,top_k}
                            How to chose the top predictions from the model for ROUGE
                            scores.
    --test_k TEST_K       The `k` parameter for the `--test_id_method`. Must be set
                            if using the `greater_k` option. (default: 3)
    --no_test_block_trigrams
                            Disable trigram blocking when calculating ROUGE scores
                            during testing. This will increase repetition and thus
                            decrease accuracy.
    --test_use_pyrouge    Use `pyrouge`, which is an interface to the official ROUGE
                            software, instead of the pure-python implementation
                            provided by `rouge-score`. You must have the real ROUGE
                            package installed. More details about ROUGE 1.5.5 here: ht
                            tps://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-
                            1.5.5. It is recommended to use this option for official
                            scores. The `ROUGE-L` measurements from `pyrouge` are
                            equivalent to the `rougeLsum` measurements from the
                            default `rouge-score` package.
    --loss_key {loss_total,loss_total_norm_batch,loss_avg_seq_sum,loss_avg_seq_mean,loss_avg}
                            Which reduction method to use with BCELoss. See the
                            `experiments/loss_functions/` folder for info on how the
                            default (`loss_avg_seq_mean`) was chosen.
