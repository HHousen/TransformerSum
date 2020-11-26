.. _pretrained_abs:

Abstractive Pre-trained Models & Results
========================================

.. important:: There are currently no pre-trained models that can be used to abstractively summarize long documents. Models listed in the :ref:`bart_converted_to_longformerencdec` section need to be fine-tuned on a long document summarization dataset, such as Arxiv-PubMed, to create a model that can summarize long sequences. The ArXiv-PubMed models will be trained as soon as the developers obtain the resources necessary to train them (2 Tesla V100 GPUs). If you have the resources to train these models, then `open a pull request <https://github.com/HHousen/TransformerSum/compare>`__.

.. _bart_converted_to_longformerencdec:

BART Converted to LongformerEncoderDecoder
------------------------------------------

.. important:: The models in this section are the raw output from the ``convert_bart_to_longformerencoderdecoder.py`` script without any gradient updates. This means that they have not been fine-tuned on any extra data and thus need to be trained.

The additional position embeddings for these models were initialized by copying the embeddings of the first ``512`` positions. This initialization is crucial for the model performance (check table 6 in `the longformer paper <https://arxiv.org/pdf/2004.05150.pdf>`_ for performance without this initialization).

The models output from the ``convert_bart_to_longformerencoderdecoder.py`` script do not work for long documents without further training. Tables 6 and 11 in `the longformer paper <https://arxiv.org/pdf/2004.05150.pdf>`_ suggest that models converted to be able to handle long content may perform well before any additional gradient updates. However, this does not appear to be true for summarization. The converted ``facebook/bart-large-cnn`` model from ``huggingface/transformers`` (aka ``longformer-encdec-bart-large-cnn-converted``) produces almost random summaries that rarely pertain to the input document. Thus, these models need to be fine-tuned on a long document summarization dataset.

These are ``huggingface/transformers`` models, so they need to be used with the ``--model_name_or_path`` option. They can also be loaded directly in ``huggingface/transformers`` using ``LongformerEncoderDecoderForConditionalGeneration.from_pretrained()``.

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
