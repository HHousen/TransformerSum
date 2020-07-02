.. _pretrained_ext:

Extractive Pre-trained Models & Results
=======================================

The recommended model to use is ``distilroberta-base-ext-sum`` because of its fast performance, relatively low number of parameters, and good performance. 

Notes
-----

The distil* models are of special significance. Distil* is a class of compressed models that started with `DistilBERT <https://arxiv.org/abs/1910.01108>`__. DistilBERT stands for Distillated-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than ``bert-base-uncased``, runs 60% faster while preserving 99% of BERT's performances as measured on the GLUE language understanding benchmark. DistilBERT is a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilRoBERTa reaches 95% of RoBERTa-base's performance on GLUE and is twice as fast as RoBERTa while being 35% smaller. More info at `huggingface/transformers <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__.

The remarkable performance to size ratio of the distil* models can be transferred to summarization. ``distilroberta`` is recommended over ``distilbert`` because of the architecture improvements that the original RoBERTa brought over the original BERT. Essentially, ``distilroberta`` is more modern than ``distilbert``.

.. important:: Interactive charts, graphs, raw data, run commands, hyperparameter choices, and more for all trained models are publicly available on the `TransformerSum Weights & Biases page <https://app.wandb.ai/hhousen/transformerextsum>`__. Please open an `issue <https://github.com/HHousen/TransformerSum/issues/new>`__ if you have questions about these models.

TODO: Add notes about 16-bit precision

.. _pretrained_ext_cnn_dm:

CNN/DM
------

+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| Name                            | Comments | Model Download                                                                                                                                                                | Data Download                                                                             |
+=================================+==========+===============================================================================================================================================================================+===========================================================================================+
| distilbert-base-uncased-ext-sum | None     | `Model <https://drive.google.com/uc?id=1-W9VzvVgKyu4d3IfNMw0k2zvXzkqpRw7>`__ & `All Checkpoints <https://drive.google.com/drive/folders/1niakD1lkqI-n2VNi21h9ugUpItc2wOnd>`__ | `CNN/DM Bert Uncased <https://drive.google.com/uc?id=100ZE4fVU73EU3K_EGktrYDoMSLJ6EUQW>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None     | `Model <https://drive.google.com/uc?id=1-2TZe28K8inHoJr2-WuVivj2qwBn7tFs>`__ & `All Checkpoints <https://drive.google.com/drive/folders/110ZO4h2MkZkD-L5_WV_PWUlVWL6QfyO6>`__ | `CNN/DM Roberta <https://drive.google.com/uc?id=1-L7UOYe69dD--OPGCa4sS0QQEnZNb_Vb>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None     | `Model <https://drive.google.com/uc?id=1TpdLPVrZ-V5X-k4pvDMDq2DdQZaFI8rw>`__ & `All Checkpoints <https://drive.google.com/drive/folders/1D2Q_9idFKPU5syWgSBWJMrP38DRJWO3U>`__ | `CNN/DM Bert Uncased <https://drive.google.com/uc?id=100ZE4fVU73EU3K_EGktrYDoMSLJ6EUQW>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None     | `Model <https://drive.google.com/uc?id=18ZlImBv1P7VmDPUpiQHF9frk-q3AFfD0>`__ & `All Checkpoints <https://drive.google.com/drive/folders/1nUzZNyYi6Lw_i8-7-e96jyEWS53ZhvJP>`__ | `CNN/DM Roberta <https://drive.google.com/uc?id=1-L7UOYe69dD--OPGCa4sS0QQEnZNb_Vb>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None     | Not yet...                                                                                                                                                                    | `CNN/DM Bert Uncased <https://drive.google.com/uc?id=100ZE4fVU73EU3K_EGktrYDoMSLJ6EUQW>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None     | Not yet...                                                                                                                                                                    | `CNN/DM Roberta <https://drive.google.com/uc?id=1-L7UOYe69dD--OPGCa4sS0QQEnZNb_Vb>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| longformer-base-4096-ext-sum    | None     | Not yet...                                                                                                                                                                    | `CNN/DM Longformer <https://drive.google.com/uc?id=1438kLkTC9zc9otkA7Q7sJqDdCxBrfWqj>`_   |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+

CNN/DM ROUGE Scores
^^^^^^^^^^^^^^^^^^^

Test set results on the CNN/DailyMail dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
| Name                            | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-L-Sum |
+=================================+============+============+============+=============+
| distilbert-base-uncased-ext-sum | 42.71      | 19.91      | 27.52      | 39.18       |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | 42.87      | 20.02      | 27.46      | 39.31       |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | 42.78      | 19.83      | 27.43      | 39.18       |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | 43.24      | 20.36      | 27.64      | 39.65       |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+

.. note:: Currently, ``distilbert`` beats ``bert-base-uncased`` by 1.0014% (``(42.71/42.78+19.91/19.83+27.52/27.43+39.18/39.18)/4=1.0014197729882865``). Since ``bert-base-uncased`` has more parameters than ``distilbert``, this is unusual and is likely a tuning issue. This suggests that tuning the hyperparameters of ``bert-base-uncased`` can improve its performance. ``distilroberta`` matches 92.7% of the performance of ``roberta-base`` (``(42.87/43.24+20.02/20.36+27.46/27.64+29.31/39.65)/4=0.9268623888753363``).

CNN/DM Training Times and Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+-------------+------------+
| Name                            | Time        | Model Size |
+=================================+=============+============+
| distilbert-base-uncased-ext-sum | 6h 22m 32s  | 796.4MB    |
+---------------------------------+-------------+------------+
| distilroberta-base-ext-sum      | 6h 21m 37s  | 980.8MB    |
+---------------------------------+-------------+------------+
| bert-base-uncased-ext-sum       | 12h 51m 17s | 1.3GB      |
+---------------------------------+-------------+------------+
| roberta-base-ext-sum            | 13h 7m 3s   | 1.5GB      |
+---------------------------------+-------------+------------+
| bert-large-uncased-ext-sum      | Not yet...  | Not yet... |
+---------------------------------+-------------+------------+
| roberta-large-ext-sum           | Not yet...  | Not yet... |
+---------------------------------+-------------+------------+

WikiHow
-------

+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| Name                            | Comments | Model Download                                                                                                                                                                | Data Download                                                                              |
+=================================+==========+===============================================================================================================================================================================+============================================================================================+
| distilbert-base-uncased-ext-sum | None     | `Model <https://drive.google.com/uc?id=1-5xfsEk8fsyJBA7638VdPHXbkEv_vWHN>`__ & `All Checkpoints <https://drive.google.com/drive/folders/1Ar8dn9cXQN_wMbzXj_vZddg1qwyVNIIv>`__ | `WikiHow Bert Uncased <https://drive.google.com/uc?id=1-IO2AgjDsJcbrmsM3R4UIRM2bMHR-Dae>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None     | `Model <https://drive.google.com/uc?id=1-79t0FvT2PBy1OubqsvY-nV3Kskt_Aem>`__ & `All Checkpoints <https://drive.google.com/drive/folders/1DhL0b7jubLvz93hbTwcCZdvTwRi5me7l>`__ | `WikiHow Roberta <https://drive.google.com/uc?id=1-aQMjCEQlKhEcimMW_WJwQusNScIT2Uf>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None     | Not yet...                                                                                                                                                                    | `WikiHow Bert Uncased <https://drive.google.com/uc?id=1-IO2AgjDsJcbrmsM3R4UIRM2bMHR-Dae>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None     | Not yet...                                                                                                                                                                    | `WikiHow Roberta <https://drive.google.com/uc?id=1-aQMjCEQlKhEcimMW_WJwQusNScIT2Uf>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None     | Not yet...                                                                                                                                                                    | `WikiHow Bert Uncased <https://drive.google.com/uc?id=1-IO2AgjDsJcbrmsM3R4UIRM2bMHR-Dae>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None     | Not yet...                                                                                                                                                                    | `WikiHow Roberta <https://drive.google.com/uc?id=1-aQMjCEQlKhEcimMW_WJwQusNScIT2Uf>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+

WikiHow ROUGE Scores
^^^^^^^^^^^^^^^^^^^^

Test set results on the WikiHow dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
| Name                            | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-L-Sum |
+=================================+============+============+============+=============+
| distilbert-base-uncased-ext-sum | 30.48      | 8.52       | 19.00      | 28.40       |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | 31.04      | 8.93       | 19.33      | 28.94       |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+

.. note:: These are the results of an abstractive model, which means they are fairly good because they come close to abstractive models. The R1/R2/RL-Sum results of a base transformer model from the `PEGASUS paper <https://arxiv.org/abs/1912.08777>`_ are 32.48/10.53/23.86. The net difference from ``distilroberta-base-ext-sum`` is +1.44/+1.6/-5.08. Compared to the **abstractive** SOTA prior to PEGASUS, which was 28.53/9.23/26.54, ``distilroberta-base-ext-sum`` performs +2.51/-0.3/+2.4. However, the base PEGASUS model obtains scores of 36.58/15.64/30.01, which are much better than ``distilroberta-base-ext-sum``, as one would expect.


WikiHow Training Times and Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+------------+------------+
| Name                            | Time       | Model Size |
+=================================+============+============+
| distilbert-base-uncased-ext-sum | 3h 44m 56s | 796.4MB    |
+---------------------------------+------------+------------+
| distilroberta-base-ext-sum      | 3h 41m 53s | 980.8MB    |
+---------------------------------+------------+------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| roberta-base-ext-sum            | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| roberta-large-ext-sum           | Not yet... | Not yet... |
+---------------------------------+------------+------------+

arXiv-PubMed
------------

+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| Name                            | Comments | Model Download | Data Download                                                                                   |
+=================================+==========+================+=================================================================================================+
| distilbert-base-uncased-ext-sum | None     | Not yet...     | `arXiv-PubMed Bert Uncased <https://drive.google.com/uc?id=17doTVEvIHr9DGesN-BmyHVz5sqWEWdEa>`_ |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None     | Not yet...     | `arXiv-PubMed Roberta <https://drive.google.com/uc?id=11pVkVO1ivC3okWq-l_xW1qQmagDE5Htt>`_      |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None     | Not yet...     | `arXiv-PubMed Bert Uncased <https://drive.google.com/uc?id=17doTVEvIHr9DGesN-BmyHVz5sqWEWdEa>`_ |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None     | Not yet...     | `arXiv-PubMed Roberta <https://drive.google.com/uc?id=11pVkVO1ivC3okWq-l_xW1qQmagDE5Htt>`_      |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None     | Not yet...     | `arXiv-PubMed Bert Uncased <https://drive.google.com/uc?id=17doTVEvIHr9DGesN-BmyHVz5sqWEWdEa>`_ |
+---------------------------------+----------+----------------+-------------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None     | Not yet...     | `arXiv-PubMed Roberta <https://drive.google.com/uc?id=11pVkVO1ivC3okWq-l_xW1qQmagDE5Htt>`_      |
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

arXiv-PubMed Training Times and Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+------------+------------+
| Name                            | Time       | Model Size |
+=================================+============+============+
| distilbert-base-uncased-ext-sum | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| distilroberta-base-ext-sum      | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| roberta-base-ext-sum            | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| roberta-large-ext-sum           | Not yet... | Not yet... |
+---------------------------------+------------+------------+
