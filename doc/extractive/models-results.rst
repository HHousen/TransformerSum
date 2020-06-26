.. _pretrained_ext:

Extractive Pre-trained Models & Results
=======================================

The recommended model to use is ``distilroberta-base-ext-sum`` because of its fast performance, relatively low number of parameters, and good performance. 

Notes
-----

The distil* models are of special significance. Distil* is a class of compressed models that started with `DistilBERT <https://arxiv.org/abs/1910.01108>`__. DistilBERT stands for Distillated-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than ``bert-base-uncased``, runs 60% faster while preserving 99% of BERT's performances as measured on the GLUE language understanding benchmark. DistilBERT is a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilRoBERTa reaches 95% of RoBERTa-base's performance on GLUE and is twice as fast as RoBERTa while being 35% smaller. More info at `huggingface/transformers <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__.

The remarkable performance to size ratio of the distil* models can be transferred to summarization. ``distilroberta`` is recommended over ``distilbert`` because of the architecture improvements that the original RoBERTa brought over the original BERT. Essentially, ``distilroberta`` is more modern than ``distilbert``.

TODO: Add notes about 16-bit precision

.. _pretrained_ext_cnn_dm:

CNN/DM
------

+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------+
| Name                            | Comments | Model Download | Data Download                                                                             |
+=================================+==========+================+===========================================================================================+
| distilbert-base-uncased-ext-sum | None     | Not yet...     | `CNN/DM Bert Uncased <https://drive.google.com/uc?id=100ZE4fVU73EU3K_EGktrYDoMSLJ6EUQW>`_ |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None     | Not yet...     | `CNN/DM Roberta <https://drive.google.com/uc?id=1-L7UOYe69dD--OPGCa4sS0QQEnZNb_Vb>`_      |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None     | Not yet...     | `CNN/DM Bert Uncased <https://drive.google.com/uc?id=100ZE4fVU73EU3K_EGktrYDoMSLJ6EUQW>`_ |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None     | Not yet...     | `CNN/DM Roberta <https://drive.google.com/uc?id=1-L7UOYe69dD--OPGCa4sS0QQEnZNb_Vb>`_      |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None     | Not yet...     | `CNN/DM Bert Uncased <https://drive.google.com/uc?id=100ZE4fVU73EU3K_EGktrYDoMSLJ6EUQW>`_ |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None     | Not yet...     | `CNN/DM Roberta <https://drive.google.com/uc?id=1-L7UOYe69dD--OPGCa4sS0QQEnZNb_Vb>`_      |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------+

CNN/DM ROUGE Scores
^^^^^^^^^^^^^^^^^^^

Test set results on the CNN/DailyMail dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
| Name                            | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-L-Sum |
+=================================+============+============+============+=============+
| distilbert-base-uncased-ext-sum | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet..   |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+

WikiHow
-------

+---------------------------------+----------+----------------+----------------------------+
| Name                            | Comments | Model Download | Data Download              |
+=================================+==========+================+============================+
| distilbert-base-uncased-ext-sum | None     | Not yet...     | `WikiHow Bert Uncased <>`_ |
+---------------------------------+----------+----------------+----------------------------+
| distilroberta-base-ext-sum      | None     | Not yet...     | `WikiHow Roberta <>`_      |
+---------------------------------+----------+----------------+----------------------------+
| bert-base-uncased-ext-sum       | None     | Not yet...     | `WikiHow Bert Uncased <>`_ |
+---------------------------------+----------+----------------+----------------------------+
| roberta-base-ext-sum            | None     | Not yet...     | `WikiHow Roberta <>`_      |
+---------------------------------+----------+----------------+----------------------------+
| bert-large-uncased-ext-sum      | None     | Not yet...     | `WikiHow Bert Uncased <>`_ |
+---------------------------------+----------+----------------+----------------------------+
| roberta-large-ext-sum           | None     | Not yet...     | `WikiHow Roberta <>`_      |
+---------------------------------+----------+----------------+----------------------------+

WikiHow ROUGE Scores
^^^^^^^^^^^^^^^^^^^^

Test set results on the WikiHow dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
| Name                            | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-L-Sum |
+=================================+============+============+============+=============+
| distilbert-base-uncased-ext-sum | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet..   |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+

arXiv-PubMed
------------

+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| Name                            | Comments | Model Download | Data Download                                                                                   |
+=================================+==========+================+=================================================================================================+
| distilbert-base-uncased-ext-sum | None     | Not yet...     | `arXiv-PubMed Bert Uncased <https://drive.google.com/uc?id=1-htznO-Io6r-9rVSTMQ1-4HYhyu21w7s>`_ |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None     | Not yet...     | `arXiv-PubMed Roberta <>`_                                                                      |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None     | Not yet...     | `arXiv-PubMed Bert Uncased <>`_                                                                 |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None     | Not yet...     | `arXiv-PubMed Roberta <>`_                                                                      |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None     | Not yet...     | `arXiv-PubMed Bert Uncased <>`_                                                                 |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None     | Not yet...     | `arXiv-PubMed Roberta <>`_                                                                      |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+

arXiv-PubMed ROUGE Scores
^^^^^^^^^^^^^^^^^^^^^^^^^

Test set results on the arXiv-PubMed dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
| Name                            | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-L-Sum |
+=================================+============+============+============+=============+
| distilbert-base-uncased-ext-sum | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet..   |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
