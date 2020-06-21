Differences from PreSumm/BertSum
================================

This project accomplishes a task similar to `BertSum <https://github.com/nlpyang/BertSum>`_/`PreSumm <https://github.com/nlpyang/PreSumm>`_ (and is based on their research). However, ``TransformerSum`` improves various aspects and adds several features on top of ``BertSum``/``PreSumm``. The most notable improvements are listed below.

.. note:: PreSumm (Yang Liu and Mirella Lapata) builds on top of BertSum (Yang Liu) by adding abstractive summarization. BertSum was extractive only.

General
-------

* This project uses ``pytorch_lighting``, which is a template to better organize PyTorch code. It automates most of the training loop. It also results in easier to read code than plain PyTorch because everything is organized.
* ``TransformerSum`` contains comments explaining design decisions and how code works. Significant functions also have detailed docstrings. See :meth:`~data.SentencesProcessor.get_features` for an example.
* Easy pre-trained model loading and modification thanks to ``pytorch_lightning``'s ``load_from_checkpoint()`` that is automatically inherited by every ``pl.LightningModule``.
* There is a ``prediction()`` function included with the :class:`extractive.ExtractiveSummarizer` and :class:`abstractive.AbstractiveSummarizer` classes that summaries a string. This makes it easy to initialize a model and quickly perform inference.

Converting a Dataset to Extractive
----------------------------------

* Separation between the dataset conversion from abstractive to extractive and the training process. This means the same converted dataset can be used by any transformer model with further automatic processing.
* The ``convert_to_extractive.py`` script uses the more up-to-date ``spacy`` whereas BertSum utilizes ``Stanford CoreNLP``. Spacy has the added benefit of being a python library (while CoreNLP is a java application), which means easier to understand code.
* Supports the same :meth:`~convert_to_extractive.greedy_selection` and :meth:`~convert_to_extractive.combination_selection` of ``BertSum``.
* A more robust CLI that allows for various desired outputs.
* Built in optional gzip compression.

Pooling for Extractive Models
-----------------------------

* Supports the ``sent_rep_tokens`` and ``mean_tokens`` pooling methods whereas ``BertSum`` only supports ``sent_rep_tokens``. See :ref:`extractive_pooling_modes` for more info.
* Pooling is separated from the main model's ``forward()`` function for easy extendability.

Data for Extractive Models
--------------------------

* The data processing and loading uses normal PyTorch ``DataLoader``s and ``Dataset``s instead of recreations of these classes as in ``BertSum``.
* A ``collate_fn`` function converts the data from the dataset to tensors suitable for input to the model, as is stand PyTorch convention.
* The ``collate_fn`` function also performs "smart batching." It performs padding on the necessary information for each batch, which is more efficient than padding for the entire dataset or each chunk.
* The :class:`data.FSIterableDataset` class loads a dataset in chunks and is a subclass of ``torch.utils.data.IterableDataset``. While ``BertSum`` also supports chunked loading to lower RAM usage, ``TransformerSum``'s technique is more robust and directly integrates with PyTorch.

Extractive Model and Training
-----------------------------

* **Compatible with every huggingface/transformers transformer encoder model.** ``BertSum`` can only use Bert, whereas this project supports all encoders by only changing two options when training.
* Easily extendable with new custom models that are saved in the ``huggingface/transformers`` format. In this way, integration with the ``longformer`` was easily accomplished (this integration was since removed since ``huggingface/transformers`` implemented the ``longformer`` directly..
* The classifier component of ``TransformerSum`` is larger (it contains two linear layers) than `BertSum` (which contains one linear layer). The additional layer was found the greatly improve performance.
* The reduction method for the BCE loss function  is different in ``TransformerSum`` than `BertSum`. `BertSum` takes the sum of the losses for each sentence (ignoring padding) even though it `looks like it uses the mean <https://github.com/nlpyang/BertSum/blob/master/src/models/trainer.py#L325>`_. Five different reduction methods were tested (see the :ref:`loss_function_experiments`). There did not appear to a significant difference, but the best was chosen.
* The batch size parameter of ``BertSum`` is not the real batch size (which is likely caused by the custom ``DataLoader``). In this project batch size is the number of documents processed on the GPU at once.
* Multiple optimizers are supported "out-of-the-box" in ``TransformerSum`` without any need to modify the code.
* The ``OneCycle`` and ``linear_schedule_with_warmup`` schedulers are supported in ``TransformerSum`` "out-of-the-box."
* Logging of all five loss functions (for both the train and validation sets), accuracy, and more is supported. Weights & Biases and Tensorboard are supported "out-of-the-box" but ``pytorch_lightning`` can integrate several other loggers.

Abstractive Model and Training
------------------------------

* Dataset preparation happens extremely quickly (minutes instead of hours; CNN/DM can be ready to train in about 10 minutes from the raw data)
* Integration with `huggingface/nlp <https://github.com/huggingface/nlp>`_ means any summarization dataset in the ``nlp`` library can be used for training by only modifying 4 options (specifically ``--dataset``, ``--dataset_version``, ``--data_example_column``, and ``--data_summarized_column``). The ``nlp`` library will handle downloading and pre-processing while the ``abstractive.py`` script will handle tokenization automatically.
* **Compatible with every huggingface/transformers EncoderDecoder model.** ``PreSumm`` only supports a BERT encoder and a standard transformer decoder, whereas this project supports all EncoderDecoder models by changing a single option (``--model_name_or_path``).

Where ``BertSum`` is Better
---------------------------

* For the extractive component, ``BertSum`` supports three classifiers: a linear layer, a transformer, and a LSTM network. This project supports three different classifiers: a few linear layers, a transformer, and a linear layer combined with a transformer. The classifier is responsible for removing the hidden features from each sentence embedding and converting them to a single number. However, the `BertSum paper <https://arxiv.org/pdf/1903.10318.pdf>`_ indicates that the difference between these classifiers is not major. ``BertSum`` has an LSTM classifier, which ``TransformerSum`` does not replicate.