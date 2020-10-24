Getting Started
===============

.. _installation_instructions:

Install
-------

Installation is made easy due to conda environments. Simply run this command from the root project directory: ``conda env create --file environment.yml`` and conda will create and environment called ``transformersum`` with all the required packages from ``environment.yml``. The spacy ``en_core_web_sm`` model is required for the ``convert_to_extractive.py`` script to detect sentence boundaries.

Step-by-Step Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone this repository: ``git clone https://github.com/HHousen/transformersum.git``.
2. Change to project directory: ``cd transformersum``.
3. Run installation command: ``conda env create --file environment.yml`` (and activate with ``conda activate transformersum``.
4. **(Optional)** If using the ``convert_to_extractive.py`` script then download the ``en_core_web_sm`` spacy model: ``python -m spacy download en_core_web_sm``.

.. _getting_started_tutorial:

Tutorial
--------

I just want to summarize some text
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GUI
~~~

If all you want to do is summarize a text string using a pre-trained model then follow the below steps:

1. Download a summarization model. Link to :ref:`pre-trained extractive models <pretrained_ext>`. Link to :ref:`pre-trained abstractive models <pretrained_abs>`.
2. Put the model in a folder named ``models`` in the project root.
3. Run ``python predictions_website.py`` and open the link. 
4. On the website enter your text, select your downloaded model, and click "SUBMIT".

Programmatically
~~~~~~~~~~~~~~~~

If you want to summarize text using a pre-trained model from python code then follow the below steps:

1. Download a summarization model. Link to :ref:`pre-trained extractive models <pretrained_ext>`. Link to :ref:`pre-trained abstractive models <pretrained_abs>`.
2. Instantiate the model:

    For extractive summarization:

    .. code-block:: python

        from extractive import ExtractiveSummarizer
        model = ExtractiveSummarizer.load_from_checkpoint("path/to/ckpt/file")

    For abstractive summarization:

    .. code-block:: python

        from abstractive import AbstractiveSummarizer
        model = AbstractiveSummarizer.load_from_checkpoint("path/to/ckpt/file")

3. Run prediction on a string of text:

    .. code-block:: python
    
        text_to_summarize = "Something Awesome"
        summary = model.predict(text_to_summarize)

.. note:: If you are using an :class:`~extractive.ExtractiveSummarizer`, then you can pass ``num_summary_sentences`` to specify the number of sentences in the output summary. For instance, ``summary = model.predict(text_to_summarize, num_summary_sentences=5)``. The default is 3 sentences. More info at :meth:`extractive.ExtractiveSummarizer.predict`.

Extractive Summarization
^^^^^^^^^^^^^^^^^^^^^^^^

Outline:

1. Convert dataset from abstractive to extractive (different for each dataset)
2. Create features and tokenize the extractive dataset (different for every model)
3. Train the model using the :ref:`training script <train_extractive_model>`
4. Test the model using the :ref:`training script <train_extractive_model>`

Lets train a model that performs extractive summarization. In this tutorial we will be using BERT, but you can easily use any `autoencoding model <https://huggingface.co/transformers/summary.html#autoencoding-models>`__ from `huggingface/transformers <https://github.com/huggingface/transformers>`__.

.. note:: Autoencoding models are pretrained by corrupting the input tokens in some way and trying to reconstruct the original sentence. They correspond to the encoder of the original transformer model in the sense that they get access to the full inputs without any mask. Those models usually build a bidirectional representation of the whole sentence. They can be fine-tuned and achieve great results on many tasks such as text generation, but their most natural application is sentence classification or token classification. A typical example of such models is BERT. For more information about the different type of transformer models go to the `Huggingface "Summary of the models" page <https://huggingface.co/transformers/summary.html>`_.

The first step to train this model is to download the data. You can see all the datasets that are natively supported on the :ref:`extractive_supported_datasets` page. We will be using the well-known :ref:`CNN/DailyMail dataset <extractive_dataset_cnn_dm>` since its summaries are relatively extractive and the input articles are not incredibly long. You can download the data from the "Data Download Link" link on the :ref:`extractive_dataset_cnn_dm` page. You can skip directly to step 2 as listed above by downloading the "Extractive Version" or you can skip to step 3 by downloading the ``bert-base-uncased-ext-sum`` model data from :ref:`pretrained_ext_cnn_dm`.

To be clear, this is an abstractive dataset so we will convert it to the extractive task using the ``convert_to_extractive.py`` script. You can read more about this script on the :ref:`convert_to_extractive` page, but in short it creates a completely extractive summary that maximizes the ROUGE score between itself and the ground-truth abstractive summary. Labels (a list of 0s and 1s where 0s correspond to sentences that should not be in the summary and 1s correspond to sentences that should be in the summary) can be generated from this extractive summary. Visit :ref:`convert_to_extractive` if you want to do this to your own dataset. For now, all you need to understand is that the above happens. You can download the preprocessed data instead of recomputing and recreating it yourself. However, there is one more step you can skip by downloading preprocessed data.

Command to convert dataset to extractive (:ref:`more info <convert_to_extractive>`):

.. code-block:: 

    python convert_to_extractive.py ./datasets/cnn_dailymail_processor/cnn_dm --shard_interval 5000 --compression --add_target_to test

Once we have an extractive dataset, we need to convert the text into features that the computer can understand. This includes ``input_ids``, ``attention_mask``, ``sent_rep_token_ids``, and more. The :meth:`extractive.ExtractiveSummarizer.forward` and :meth:`data.SentencesProcessor.get_features` docstrings explains these features nicely. The `huggingface/transformers glossary <https://huggingface.co/transformers/glossary.html>`_ is a good resource as well. This conversion to model-specific features happens automatically before training begins. Since the features are model-specific, the training script is responsible for converting the data. It creates a :class:`~data.SentencesProcessor` that does most of the heavy lifting. You can learn more about this automatic preprocessing on the :ref:`data_automatic_preprocessing` page. 

Command to only pre-process the data and stop right before training would begin (:ref:`more info <data_automatic_preprocessing>`):

.. code-block:: 

    python main.py --data_path ./datasets/cnn_dailymail_processor/cnn_dm --use_logger tensorboard --model_name_or_path bert-base-uncased --model_type bert --do_train --only_preprocess

If you didn't run the above commands then download the ``bert-base-uncased-ext-sum`` model data from :ref:`pretrained_ext_cnn_dm`. You can do this from the command line with ``gdown <link_to_data>`` (install ``gdown`` with ``pip install gdown``). Extract the data with ``tar -xzvf bert-base-uncased.tar.gz``. Now you are ready to train. The BERT model will be downloaded automatically by the ``huggingface/transformers`` library.

Training command:

.. code-block:: 

    python main.py \
    --model_name_or_path bert-base-uncased \
    --model_type bert \
    --data_path ./bert-base-uncased \
    --max_epochs 3 \
    --accumulate_grad_batches 2 \
    --warmup_steps 2300 \
    --gradient_clip_val 1.0 \
    --optimizer_type adamw \
    --use_scheduler linear \
    --do_train --do_test \
    --batch_size 16

You can learn more about the above command on :ref:`train_extractive_model`. 

Abstractive Summarization
^^^^^^^^^^^^^^^^^^^^^^^^^

Lets train a model that performs abstractive summarization. Whereas autoencoding models are used for extractive summarization, sequence-to-sequence (seq2seq) models are used for abstractive summarization. In short, autoregressive models correspond to the decoder of the original transformer model, autoencoding models correspond to the encoder, and sequence-to-sequence models use both the encoder and the decoder of the original transformer. 

.. note:: Sequence-to-sequence models use both the encoder and the decoder of the original transformer, either for translation tasks or by transforming other tasks to sequence-to-sequence problems. They can be fine-tuned to many tasks but their most natural applications are translation, summarization and question answering. The original transformer model is an example of such a model (only for translation), T5 is an example that can be fine-tuned on other tasks.

You can easily fine-tune a seq2seq model on a summarization dataset using the `summarization examples in huggingface/transformers <https://github.com/huggingface/transformers/tree/master/examples/seq2seq>`_. Thus, in this project we focus on being able to use any autoencoding model with a autoregressive model to create an `EncoderDecoderModel <https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoderdecodermodel>`_. We also focus on performing :ref:`abstractive summarization on long sequences <abstractive_long_summarization>` (or :ref:`see the below short explanation <getting_started_long_abs_summarization>`).

In this tutorial we will be constructing bert-to-bert, but you can easily use a different model combination from `huggingface/transformers <https://github.com/huggingface/transformers>`__. The ``--model_name_or_path`` option specifies the encoder and the ``--decoder_model_name_or_path`` specifies the decoder. If ``--decoder_model_name_or_path`` is not set then the value of ``--model_name_or_path`` is used for the decoder.

Any summarization dataset from `huggingface/nlp <https://github.com/huggingface/nlp>`_ can be used for training by only changing 4 options (specifically ``--dataset``, ``--dataset_version``, ``--data_example_column``, and ``--data_summarized_column``). The ``nlp`` library will handle downloading and pre-processing while the ``abstractive.py`` script will handle tokenization automatically. The `CNN/DM dataset <https://huggingface.co/nlp/viewer/?dataset=cnn_dailymail&config=3.0.0>`__ is the default so if you want to use that dataset you don't need to specify any options concerning data. There is a list of suggested datasets at :ref:`abstractive_supported_datasets`.

So, in brief, training an abstractive model is as easy as running one command. Go to :ref:`abstractive_command_example` for an example training command.

**Long Sequences Abstractive - Quick Tutorial:** Create ``LongformerEncoderDecoder`` using the directions at :ref:`abstractive_long_summarization` or download one from :ref:`bart_converted_to_longformerencdec`. Then, use the path to ``LongformerEncoderDecoder`` as the ``--model_name_or_path``. The path to ``LongformerEncoderDecoder`` must contain "longformer-encdec". You can now create summaries from sequences up to 4096 tokens (or up to 16,000 tokens depending on the ``max_pos`` value used).

Long Sequence Summarization
---------------------------

This project can summarize long sequences (where long sequences are considered those greater than 512-1024 tokens) using both extractive and abstractive models.

To perform **extractive summarization** on long sequences, simply use the ``longformer`` model as the ``word_embedding_model``, which is specified by ``--model_name_or_path``. In other words, set ``--model_name_or_path`` to ``allenai/longformer-base-4096`` or ``allenai/longformer-large-4096`` to summarize documents of max length 4,096. For the most up-to-date model shortcut codes visit the `huggingface pretrained models page <https://huggingface.co/transformers/pretrained_models.html>`_ and the `community models page <https://huggingface.co/models>`_.

.. _getting_started_long_abs_summarization:

For **abstractive summarization** the setup is a little more complicated. Abstractive text summarization is a sequence-to-sequence problem solved by `sequence-to-sequence models <https://huggingface.co/transformers/summary.html#sequence-to-sequence-models>`_. However, state-of-the-art seq2seq models only function on short sequences. However, BART can be modified to use the sliding window attention from the longformer to create a seq2seq model that can abstractively summarize sequences up to 16,000 tokens. Visit :ref:`abstractive_long_summarization` for more information.
