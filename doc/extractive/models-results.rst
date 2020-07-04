.. _pretrained_ext:

Extractive Pre-trained Models & Results
=======================================

The recommended model to use is ``distilroberta-base-ext-sum`` because of its fast performance, relatively low number of parameters, and good performance. 

Notes
-----

The distil* models are of special significance. Distil* is a class of compressed models that started with `DistilBERT <https://arxiv.org/abs/1910.01108>`__. DistilBERT stands for Distillated-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than ``bert-base-uncased``, runs 60% faster while preserving 99% of BERT's performances as measured on the GLUE language understanding benchmark. DistilBERT is a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilRoBERTa reaches 95% of RoBERTa-base's performance on GLUE and is twice as fast as RoBERTa while being 35% smaller. More info at `huggingface/transformers <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__.

The remarkable performance to size ratio of the distil* models can be transferred to summarization. ``distilroberta`` is recommended over ``distilbert`` because of the architecture improvements that the original RoBERTa brought over the original BERT. Essentially, ``distilroberta`` is more modern than ``distilbert``.

`MobileBERT <https://arxiv.org/abs/2004.02984>`_ is similar to ``distilbert`` in that it is a smaller version of BERT that achieves amazing performance at a very small size. `According to the authors <https://openreview.net/forum?id=SJxjVaNKwB&noteId=S1gxqk_7jH>`__, MobileBERT is *2.64x smaller and 2.45x faster* than DistilBERT. DistilBERT successfully halves the depth of BERT model by knowledge distillation in the pre-training stage and an optional fine-tuning stage. MobileBERT only uses knowledge transfer in the pre-training stage and does not require a fine-tuned teacher or data augmentation in the down-stream tasks. DistilBERT compresses BERT by reducing its depth, while MobileBERT compresses BERT by reducing its width, which has been shown to be more effective.

.. important:: Interactive charts, graphs, raw data, run commands, hyperparameter choices, and more for all trained models are publicly available on the `TransformerSum Weights & Biases page <https://app.wandb.ai/hhousen/transformerextsum>`__. Please open an `issue <https://github.com/HHousen/TransformerSum/issues/new>`__ if you have questions about these models.

Additionally, all of the models on this page were trained completely for free using Tesla P100-PCIE-16GB GPUs on `Google Colaboratory <https://colab.research.google.com/>`_. Those that took over 12 hours to train were split into multiple training sessions since ``pytorch_lightning`` enables easy resuming with the ``--resume_from_checkpoint`` argument.

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
| distilbert-base-uncased-ext-sum | None     | `Model <https://drive.google.com/uc?id=1-2Kjziq7hU4k0zMTlE26FjFyCc_A63xq>`__ & `All Checkpoints <https://drive.google.com/drive/folders/1Ar8dn9cXQN_wMbzXj_vZddg1qwyVNIIv>`__ | `WikiHow Bert Uncased <https://drive.google.com/uc?id=1-IO2AgjDsJcbrmsM3R4UIRM2bMHR-Dae>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None     | `Model <https://drive.google.com/uc?id=1-3NV3TdRcTta9JTi9Kh0sWtoNLEdWrY1>`__ & `All Checkpoints <https://drive.google.com/drive/folders/1DhL0b7jubLvz93hbTwcCZdvTwRi5me7l>`__ | `WikiHow Roberta <https://drive.google.com/uc?id=1-aQMjCEQlKhEcimMW_WJwQusNScIT2Uf>`_      |
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
| distilbert-base-uncased-ext-sum | 30.69      | 8.65       | 19.13      | 28.58       |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | 31.07      | 8.96       | 19.34      | 28.95       |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+

.. note:: These are the results of an extractive model, which means they are fairly good because they come close to abstractive models. The R1/R2/RL-Sum results of a base transformer model from the `PEGASUS paper <https://arxiv.org/abs/1912.08777>`_ are 32.48/10.53/23.86. The net difference from ``distilroberta-base-ext-sum`` is +1.41/+1.57/-5.09. Compared to the **abstractive** SOTA prior to PEGASUS, which was 28.53/9.23/26.54, ``distilroberta-base-ext-sum`` performs +2.54/-0.27/+2.41. However, the base PEGASUS model obtains scores of 36.58/15.64/30.01, which are much better than ``distilroberta-base-ext-sum``, as one would expect.


WikiHow Training Times and Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+------------+------------+
| Name                            | Time       | Model Size |
+=================================+============+============+
| distilbert-base-uncased-ext-sum | 3h 42m 12s | 796.4MB    |
+---------------------------------+------------+------------+
| distilroberta-base-ext-sum      | 4h 27m 23s | 980.8MB    |
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

+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| Name                            | Comments | Model Download                                                                                                                                                                | Data Download                                                                                   |
+=================================+==========+===============================================================================================================================================================================+=================================================================================================+
| distilbert-base-uncased-ext-sum | None     | Not yet...                                                                                                                                                                    | `arXiv-PubMed Bert Uncased <https://drive.google.com/uc?id=17doTVEvIHr9DGesN-BmyHVz5sqWEWdEa>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None     | `Model <https://drive.google.com/uc?id=1-8xVR72-jWtIxvl6DYvcND2yVc0gxjGR>`__ & `All Checkpoints <https://drive.google.com/drive/folders/1jNWCOa8bxNh_AEKJ42-LeC6H5tZhWB8p>`__ | `arXiv-PubMed Roberta <https://drive.google.com/uc?id=11pVkVO1ivC3okWq-l_xW1qQmagDE5Htt>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None     | Not yet...                                                                                                                                                                    | `arXiv-PubMed Bert Uncased <https://drive.google.com/uc?id=17doTVEvIHr9DGesN-BmyHVz5sqWEWdEa>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None     | Not yet...                                                                                                                                                                    | `arXiv-PubMed Roberta <https://drive.google.com/uc?id=11pVkVO1ivC3okWq-l_xW1qQmagDE5Htt>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None     | Not yet...                                                                                                                                                                    | `arXiv-PubMed Bert Uncased <https://drive.google.com/uc?id=17doTVEvIHr9DGesN-BmyHVz5sqWEWdEa>`_ |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None     | Not yet...                                                                                                                                                                    | `arXiv-PubMed Roberta <https://drive.google.com/uc?id=11pVkVO1ivC3okWq-l_xW1qQmagDE5Htt>`_      |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+
| longformer-base-4096-ext-sum    | None     | Not yet...                                                                                                                                                                    | `arXiv-PubMed Longformer <https://drive.google.com/uc?id=17IEoiKzs_XO1xo4mQTTcHNGhUsTxbn4G>`_   |
+---------------------------------+----------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------+

arXiv-PubMed ROUGE Scores
^^^^^^^^^^^^^^^^^^^^^^^^^

Test set results on the arXiv-PubMed dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
| Name                            | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-L-Sum |
+=================================+============+============+============+=============+
| distilbert-base-uncased-ext-sum | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | 34.70      | 12.16      | 19.52      | 30.82       |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+

.. note:: These are the results of an extractive model, which means they are fairly good because they come close to abstractive models. The R1/R2/RL-Sum results of a base transformer model from the `PEGASUS paper <https://arxiv.org/abs/1912.08777>`_ are 34.79/7.69/19.51 (average of 35.63/7.95/20.00 (arXiv) and 33.94/7.43/19.02 (PubMed)). The net difference from ``distilroberta-base-ext-sum`` is +0.09/-4.47/-11.31. Compared to the **abstractive** SOTA prior to PEGASUS, which was 41.09/14.93/23.57 (average of 41.59/14.26/23.55 (arXiv) and 40.59/15.59/23.59 (PubMed)), ``distilroberta-base-ext-sum`` performs -6.39/-2.77/+7.25. However, the base PEGASUS model obtains scores of 37.39/12.66/23.87 (average of 34.81/10.16/22.50 (arXiv) and 39.98/15.15/25.23 (PubMed)). The large model obtains scores of 45.10/18.59/26.75 (average of 44.70/17.27/25.80 (arXiv) and 45.49/19.90/27.69 (PubMed)) which are much better than ``distilroberta-base-ext-sum``, as one would expect.

arXiv-PubMed Training Times and Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+------------+------------+
| Name                            | Time       | Model Size |
+=================================+============+============+
| distilbert-base-uncased-ext-sum | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| distilroberta-base-ext-sum      | 6h 33m 58s | 980.8MB    |
+---------------------------------+------------+------------+
| bert-base-uncased-ext-sum       | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| roberta-base-ext-sum            | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| roberta-large-ext-sum           | Not yet... | Not yet... |
+---------------------------------+------------+------------+
