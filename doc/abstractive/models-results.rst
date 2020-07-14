.. _pretrained_abs:

Abstractive Pre-trained Models & Results
========================================

.. _bart_converted_to_longformerencdec:

BART Converted to LongformerEncoderDecoder
------------------------------------------

These models are the raw output from the ``convert_bart_to_longformerencoderdecoder.py`` script without any gradient updates. This means that they have not been finetuned on any extra data. The additional position embeddings were initialized by copying the embeddings of the first ``512`` positions. This initialization is crucial for the model performance (check table 6 in `the longformer paper <https://arxiv.org/pdf/2004.05150.pdf>`_ for performance without this initialization.

The output of the ``convert_bart_to_longformerencoderdecoder.py`` script works for long documents even without further training. Check tables 6 and 11 in `the longformer paper <https://arxiv.org/pdf/2004.05150.pdf>`_ to get a sense of the expected performance of this model before any additional gradient updates.

The Google Drive folder containing all of the below models is available `at this link <https://drive.google.com/drive/folders/18W_wJ5ovpSN98AT4626JtEJUCW_wA4ax>`__.

+--------------------------------------------------+----------+------------------------------------------------------------------------------------------------------+
| Name                                             | Comments | Model Size Download                                                                                  |
+==================================================+==========+======================================================================================================+
| longformer-encdec-bart-base-converted            | None     | `Model 4096/8192/12288 <https://drive.google.com/drive/folders/1DBxRZkOHS7OdU80L8OvnzCa3K6-ho_Dj>`__ |
+--------------------------------------------------+----------+------------------------------------------------------------------------------------------------------+
| longformer-encdec-bart-large-converted           | None     | `Model 4096/8192/12288 <https://drive.google.com/drive/folders/10gPiqlAdIART4cMNWhI1fIVZ3J1L5hAW>`__ |
+--------------------------------------------------+----------+------------------------------------------------------------------------------------------------------+
| longformer-encdec-bart-large-cnn-converted       | None     | `Model 4096/8192/12288 <https://drive.google.com/drive/folders/12T_M5xlApGv6SCQSoMQpGO3sqEv5r_kW>`__ |
+--------------------------------------------------+----------+------------------------------------------------------------------------------------------------------+
| longformer-encdec-distilbart-cnn-12-6-converted  | None     | `Model 4096/8192/12288 <https://drive.google.com/drive/folders/13hoepJXqCxRF621pritYtiGw_Fn2qST0>`__ |
+--------------------------------------------------+----------+------------------------------------------------------------------------------------------------------+
| longformer-encdec-distilbart-xsum-12-6-converted | None     | `Model 4096/8192/12288 <https://drive.google.com/drive/folders/14yDp-yncJuFSjhxOQuw07YKdcDdu9ELU>`__ |
+--------------------------------------------------+----------+------------------------------------------------------------------------------------------------------+

arXiv-PubMed
------------

+----------------------------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| Name                                               | Comments | Model Download                                                                                              | Data Download |
+====================================================+==========+=============================================================================================================+===============+
| bart-base-abs-sum                                  | None     | `Model <https://drive.google.com/uc?id=>`__ & `All Checkpoints <https://drive.google.com/drive/folders/>`__ | Not yet..     |
+----------------------------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| bart-large-abs-sum                                 | None     | Not yet...                                                                                                  | Not yet..     |
+----------------------------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| longformer-encdec-8192-bart-large-abs-sum          | None     | Not yet...                                                                                                  | Not yet..     |
+----------------------------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| longformer-encdec-8192-bart-base-abs-sum           | None     | Not yet...                                                                                                  | Not yet..     |
+----------------------------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| longformer-encdec-8192-distilbart-cnn-12-6-abs-sum | None     | Not yet...                                                                                                  | Not yet..     |
+----------------------------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+

arXiv-PubMed ROUGE Scores
^^^^^^^^^^^^^^^^^^^^^^^^^

Test set results on the arXiv-PubMed dataset using ROUGE F\ :sub:`1`\ .

+----------------------------------------------------+------------+------------+------------+-------------+
| Name                                               | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-L-Sum |
+====================================================+============+============+============+=============+
| bart-base-abs-sum                                  | Not yet... | Not yet... | Not yet... | Not yet...  |
+----------------------------------------------------+------------+------------+------------+-------------+
| bart-large-abs-sum                                 | Not yet... | Not yet... | Not yet... | Not yet...  |
+----------------------------------------------------+------------+------------+------------+-------------+
| longformer-encdec-8192-bart-large-abs-sum          | Not yet... | Not yet... | Not yet... | Not yet...  |
+----------------------------------------------------+------------+------------+------------+-------------+
| longformer-encdec-8192-bart-base-abs-sum           | Not yet... | Not yet... | Not yet... | Not yet...  |
+----------------------------------------------------+------------+------------+------------+-------------+
| longformer-encdec-8192-distilbart-cnn-12-6-abs-sum | Not yet... | Not yet... | Not yet... | Not yet...  |
+----------------------------------------------------+------------+------------+------------+-------------+
| longformer-encdec-bart-large-cnn-converted         | Not yet... | Not yet... | Not yet... | Not yet...  |
+----------------------------------------------------+------------+------------+------------+-------------+

.. note:: ``longformer-encdec-bart-large-cnn-converted`` is the same as the model from :ref:`bart_converted_to_longformerencdec`. It was only tested and not trained on the arXiv-PubMed dataset.
