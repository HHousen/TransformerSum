.. _pretrained_ext:

Extractive Pre-trained Models & Results
=======================================

The recommended model to use is ``distilroberta-base-ext-sum`` because of its fast performance, relatively low number of parameters, and good performance.

Notes
-----

The distil* models are of special significance. Distil* is a class of compressed models that started with `DistilBERT <https://arxiv.org/abs/1910.01108>`__. DistilBERT stands for Distillated-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than ``bert-base-uncased``, runs 60% faster while preserving 99% of BERT's performances as measured on the GLUE language understanding benchmark. DistilBERT is a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilRoBERTa reaches 95% of RoBERTa-base's performance on GLUE and is twice as fast as RoBERTa while being 35% smaller. More info at `huggingface/transformers <https://github.com/huggingface/transformers/tree/master/examples/distillation>`__.

The remarkable performance to size ratio of the distil* models can be transferred to summarization. ``distilroberta`` is recommended over ``distilbert`` because of the architecture improvements that the original RoBERTa brought over the original BERT. Essentially, ``distilroberta`` is more modern than ``distilbert``.

`MobileBERT <https://arxiv.org/abs/2004.02984>`_ is similar to ``distilbert`` in that it is a smaller version of BERT that achieves amazing performance at a very small size. `According to the authors <https://openreview.net/forum?id=SJxjVaNKwB&noteId=S1gxqk_7jH>`__, MobileBERT is *2.64x smaller and 2.45x faster* than DistilBERT. DistilBERT successfully halves the depth of BERT model by knowledge distillation in the pre-training stage and an optional fine-tuning stage. MobileBERT only uses knowledge transfer in the pre-training stage and does not require a fine-tuned teacher or data augmentation in the down-stream tasks. DistilBERT compresses BERT by reducing its depth, while MobileBERT compresses BERT by reducing its width, which has been shown to be more effective. MobileBERT usually needs a larger learning rate and more training epochs in fine-tuning than the original BERT.

.. important:: Interactive charts, graphs, raw data, run commands, hyperparameter choices, and more for all trained models are publicly available on the `TransformerSum Weights & Biases page <https://app.wandb.ai/hhousen/transformerextsum>`__. You can download the raw data for each model on this site, or `download an overview as a CSV <../_static/summarization-model-experiments-raw-data.csv>`__. Please open an `issue <https://github.com/HHousen/TransformerSum/issues/new>`__ if you have questions about these models.

Additionally, all of the models on this page were trained completely for free using Tesla P100-PCIE-16GB GPUs on `Google Colaboratory <https://colab.research.google.com/>`_. Those that took over 12 hours to train were split into multiple training sessions since ``pytorch_lightning`` enables easy resuming with the ``--resume_from_checkpoint`` argument.

.. _pretrained_ext_cnn_dm:

CNN/DM
------

+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
|              Name               |       Comments       |                                                      Model Download                                                      |                                                                        Data Download                                                                        |
+=================================+======================+==========================================================================================================================+=============================================================================================================================================================+
| distilbert-base-uncased-ext-sum | None                 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/models/distilbert-base-uncased/epoch=3.ckpt>`__   | `CNN/DM Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_     |
+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None                 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/models/distilroberta-base/epoch=3.ckpt>`__        | `CNN/DM Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                    |
+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None                 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/models/bert-base-uncased/epoch=3.ckpt>`__         | `CNN/DM Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_     |
+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None                 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/models/roberta-base/epoch=3.ckpt>`__              | `CNN/DM Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                    |
+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None                 | Not yet...                                                                                                               | `CNN/DM Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_     |
+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None                 | Not yet...                                                                                                               | `CNN/DM Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                    |
+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| longformer-base-4096-ext-sum    | None                 | Not yet...                                                                                                               | `CNN/DM Longformer <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/longformer-base-4096/longformer-base-4096.tar.gz>`_ |
+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
| mobilebert-uncased-ext-sum      | Trained with lr=8e-5 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/models/mobilebert-uncased_8e-5/epoch%3D3.ckpt>`__ | `CNN/DM Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_     |
+---------------------------------+----------------------+--------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+

CNN/DM ROUGE Scores
^^^^^^^^^^^^^^^^^^^

Test set results on the CNN/DailyMail dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
|              Name               |  ROUGE-1   |  ROUGE-2   |  ROUGE-L   | ROUGE-L-Sum |
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
| longformer-base-4096-ext-sum    | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| mobilebert-uncased-ext-sum      | 42.01      | 19.31      | 26.89      | 38.53       |
+---------------------------------+------------+------------+------------+-------------+

.. note:: Currently, ``distilbert`` beats ``bert-base-uncased`` by 1.0014% (``(42.71/42.78+19.91/19.83+27.52/27.43+39.18/39.18)/4=1.0014197729882865``). Since ``bert-base-uncased`` has more parameters than ``distilbert``, this is unusual and is likely a tuning issue. This suggests that tuning the hyperparameters of ``bert-base-uncased`` can improve its performance. ``distilroberta`` matches 92.7% of the performance of ``roberta-base`` (``(42.87/43.24+20.02/20.36+27.46/27.64+29.31/39.65)/4=0.9268623888753363``).

.. important:: ``mobilebert-uncased-ext-sum`` achieves 96.59% (``(42.01/43.25+19.31/20.24+38.53/39.63)/3``) of the performance of `BertSum <https://arxiv.org/abs/1903.10318>`_ while containing 4.45 times (``109483009/24582401``) fewer parameters. It achieves 94.06% (``(42.01/44.41+19.31/20.86+38.53/40.55)/3``) of the performance of `MatchSum (Zhong et al., 2020) <https://arxiv.org/abs/2004.08795>`_, the current extractive state-of-the-art.

CNN/DM Training Times and Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+-------------+------------+
|              Name               |    Time     | Model Size |
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
| longformer-base-4096-ext-sum    | Not yet...  | Not yet... |
+---------------------------------+-------------+------------+
| mobilebert-uncased-ext-sum      | 8h 26m 32s  | 295.6MB    |
+---------------------------------+-------------+------------+

.. important:: ``distilroberta-base-ext-sum`` trains in about 6.5 hours on 1 P100-PCIE-16GB GPU, while `MatchSum <https://arxiv.org/abs/2004.08795>`_, the current state-of-the-art in extractive summarization on CNN/DM, takes 30 hours on 8 Tesla-V100-16G GPUs to train. If a V100 is about 2x as powerful as a P100, then it would take 480 hours (``30*8*2``) to train MatchSum on one P100. This simplistic approximation suggests that it takes about 74x (``480/6.5``) more time to train MatchSum than ``distilroberta-base-ext-sum``.

WikiHow
-------

+---------------------------------+----------------------+-------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
|              Name               |       Comments       |                                                     Model Download                                                      |                                                                       Data Download                                                                        |
+=================================+======================+=========================================================================================================================+============================================================================================================================================================+
| distilbert-base-uncased-ext-sum | None                 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/models/distilbert-base-uncased/epoch=3.ckpt>`__ | `WikiHow Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_ |
+---------------------------------+----------------------+-------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None                 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/models/distilroberta-base/epoch=3.ckpt>`__      | `WikiHow Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                |
+---------------------------------+----------------------+-------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None                 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/models/bert-base-uncased/epoch=2.ckpt>`__       | `WikiHow Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_ |
+---------------------------------+----------------------+-------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None                 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/models/roberta-base/epoch=2.ckpt>`__            | `WikiHow Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                |
+---------------------------------+----------------------+-------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None                 | Not yet...                                                                                                              | `WikiHow Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_ |
+---------------------------------+----------------------+-------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None                 | Not yet...                                                                                                              | `WikiHow Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                |
+---------------------------------+----------------------+-------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| mobilebert-uncased-ext-sum      | Trained with lr=8e-5 | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/models/mobilebert-uncased/epoch=3.ckpt>`__      | `WikiHow Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_ |
+---------------------------------+----------------------+-------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+

WikiHow ROUGE Scores
^^^^^^^^^^^^^^^^^^^^

Test set results on the WikiHow dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
|              Name               |  ROUGE-1   |  ROUGE-2   |  ROUGE-L   | ROUGE-L-Sum |
+=================================+============+============+============+=============+
| distilbert-base-uncased-ext-sum | 30.69      | 8.65       | 19.13      | 28.58       |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | 31.07      | 8.96       | 19.34      | 28.95       |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | 30.68      | 08.67      | 19.16      | 28.59       |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | 31.26      | 09.09      | 19.47      | 29.14       |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| mobilebert-uncased-ext-sum      | 30.72      | 8.78       | 19.18      | 28.59       |
+---------------------------------+------------+------------+------------+-------------+

.. note:: These are the results of an extractive model, which means they are fairly good because they come close to abstractive models. The R1/R2/RL-Sum results of a base transformer model from the `PEGASUS paper <https://arxiv.org/abs/1912.08777>`_ are 32.48/10.53/23.86. The net difference from ``distilroberta-base-ext-sum`` is +1.41/+1.57/-5.09. Compared to the **abstractive** SOTA prior to PEGASUS, which was 28.53/9.23/26.54, ``distilroberta-base-ext-sum`` performs +2.54/-0.27/+2.41. However, the base PEGASUS model obtains scores of 36.58/15.64/30.01, which are much better than ``distilroberta-base-ext-sum``, as one would expect.


WikiHow Training Times and Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+------------+------------+
|              Name               |    Time    | Model Size |
+=================================+============+============+
| distilbert-base-uncased-ext-sum | 3h 42m 12s | 796.4MB    |
+---------------------------------+------------+------------+
| distilroberta-base-ext-sum      | 4h 27m 23s | 980.8MB    |
+---------------------------------+------------+------------+
| bert-base-uncased-ext-sum       | 7h 29m 06s | 1.3GB      |
+---------------------------------+------------+------------+
| roberta-base-ext-sum            | 7h 35m 59s | 1.5GB      |
+---------------------------------+------------+------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| roberta-large-ext-sum           | Not yet... | Not yet... |
+---------------------------------+------------+------------+
| mobilebert-uncased-ext-sum      | 4h 22m 19s | 295.6MB    |
+---------------------------------+------------+------------+

arXiv-PubMed
------------

+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|              Name               | Comments |                                                        Model Download                                                        |                                                                                 Data Download                                                                                 |
+=================================+==========+==============================================================================================================================+===============================================================================================================================================================================+
| distilbert-base-uncased-ext-sum | None     | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/models/distilbert-base-uncased/epoch=3.ckpt>`__ | `arXiv-PubMed Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_     |
+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| distilroberta-base-ext-sum      | None     | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/models/distilroberta-base/epoch=3.ckpt>`__      | `arXiv-PubMed Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                    |
+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| bert-base-uncased-ext-sum       | None     | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/models/bert-base-uncased/epoch=2.ckpt>`__       | `arXiv-PubMed Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_     |
+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| roberta-base-ext-sum            | None     | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/models/roberta-base/epoch=2.ckpt>`__            | `arXiv-PubMed Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                    |
+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| bert-large-uncased-ext-sum      | None     | Not yet...                                                                                                                   | `arXiv-PubMed Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_     |
+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| roberta-large-ext-sum           | None     | Not yet...                                                                                                                   | `arXiv-PubMed Roberta <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/roberta-base/roberta-base.tar.gz>`_                    |
+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| longformer-base-4096-ext-sum    | None     | Not yet...                                                                                                                   | `arXiv-PubMed Longformer <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/longformer-base-4096/longformer-base-4096.tar.gz>`_ |
+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| mobilebert-uncased-ext-sum      | None     | `Model <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/models/mobilebert-uncased/epoch=3.ckpt>`__      | `arXiv-PubMed Bert Uncased <https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz>`_     |
+---------------------------------+----------+------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

arXiv-PubMed ROUGE Scores
^^^^^^^^^^^^^^^^^^^^^^^^^

Test set results on the arXiv-PubMed dataset using ROUGE F\ :sub:`1`\ .

+---------------------------------+------------+------------+------------+-------------+
|              Name               |  ROUGE-1   |  ROUGE-2   |  ROUGE-L   | ROUGE-L-Sum |
+=================================+============+============+============+=============+
| distilbert-base-uncased-ext-sum | 34.93      | 12.21      | 19.62      | 31.00       |
+---------------------------------+------------+------------+------------+-------------+
| distilroberta-base-ext-sum      | 34.70      | 12.16      | 19.52      | 30.82       |
+---------------------------------+------------+------------+------------+-------------+
| bert-base-uncased-ext-sum       | 34.80      | 12.26      | 19.67      | 30.92       |
+---------------------------------+------------+------------+------------+-------------+
| roberta-base-ext-sum            | 34.81      | 12.26      | 19.65      | 30.91       |
+---------------------------------+------------+------------+------------+-------------+
| bert-large-uncased-ext-sum      | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| roberta-large-ext-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| longformer-base-4096-ext-sum    | Not yet... | Not yet... | Not yet... | Not yet...  |
+---------------------------------+------------+------------+------------+-------------+
| mobilebert-uncased-ext-sum      | 33.97      | 11.74      | 19.63      | 30.19       |
+---------------------------------+------------+------------+------------+-------------+

.. note:: These are the results of an extractive model, which means they are fairly good because they come close to abstractive models. The R1/R2/RL-Sum results of a base transformer model from the `PEGASUS paper <https://arxiv.org/abs/1912.08777>`_ are 34.79/7.69/19.51 (average of 35.63/7.95/20.00 (arXiv) and 33.94/7.43/19.02 (PubMed)). The net difference from ``distilroberta-base-ext-sum`` is +0.09/-4.47/-11.31. Compared to the **abstractive** SOTA prior to PEGASUS, which was 41.09/14.93/23.57 (average of 41.59/14.26/23.55 (arXiv) and 40.59/15.59/23.59 (PubMed)), ``distilroberta-base-ext-sum`` performs -6.39/-2.77/+7.25. However, the base PEGASUS model obtains scores of 37.39/12.66/23.87 (average of 34.81/10.16/22.50 (arXiv) and 39.98/15.15/25.23 (PubMed)). The large model obtains scores of 45.10/18.59/26.75 (average of 44.70/17.27/25.80 (arXiv) and 45.49/19.90/27.69 (PubMed)) which are much better than ``distilroberta-base-ext-sum``, as one would expect.

arXiv-PubMed Training Times and Model Sizes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------------+-------------+------------+
|              Name               |    Time     | Model Size |
+=================================+=============+============+
| distilbert-base-uncased-ext-sum | 06h 46m 0s  | 796.4MB    |
+---------------------------------+-------------+------------+
| distilroberta-base-ext-sum      | 06h 33m 58s | 980.8MB    |
+---------------------------------+-------------+------------+
| bert-base-uncased-ext-sum       | 14h 40m 10s | 1.3GB      |
+---------------------------------+-------------+------------+
| roberta-base-ext-sum            | 14h 39m 43s | 1.5GB      |
+---------------------------------+-------------+------------+
| bert-large-uncased-ext-sum      | Not yet...  | Not yet... |
+---------------------------------+-------------+------------+
| roberta-large-ext-sum           | Not yet...  | Not yet... |
+---------------------------------+-------------+------------+
| longformer-base-4096-ext-sum    | Not yet...  | Not yet... |
+---------------------------------+-------------+------------+
| mobilebert-uncased-ext-sum      | 09h 5m 45s  | 295.6MB    |
+---------------------------------+-------------+------------+
