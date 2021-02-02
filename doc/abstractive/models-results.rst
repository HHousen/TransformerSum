.. _pretrained_abs:

Abstractive Pre-trained Models & Results
========================================

.. _bart_converted_to_longformerencdec:

BART Converted to LongformerEncoderDecoder
------------------------------------------

.. important:: The models in this section are the output from the `convert_bart_to_longformerencoderdecoder.py script <https://github.com/allenai/longformer/blob/master/scripts/convert_bart_to_longformerencoderdecoder.py>`_ without any gradient updates. This means that they need to be fine-tuned on a long document summarization dataset, such as Arxiv-PubMed, in order to create a model that can summarize long sequences.

The additional position embeddings for these models were initialized by copying the embeddings of the first ``512`` positions. This initialization is crucial for the model performance (check table 6 in `the longformer paper <https://arxiv.org/pdf/2004.05150.pdf>`_ for performance without this initialization).

The models output from the ``convert_bart_to_longformerencoderdecoder.py`` script do not work for long documents without further training. Tables 6 and 11 in `the longformer paper <https://arxiv.org/pdf/2004.05150.pdf>`_ suggest that models converted to be able to handle long content may perform well before any additional gradient updates. However, this does not appear to be true for summarization. The converted ``facebook/bart-large-cnn`` model from ``huggingface/transformers`` (aka ``longformer-encdec-bart-large-cnn-converted``) produces almost random summaries that rarely pertain to the input document. Thus, these models need to be fine-tuned on a long document summarization dataset.

These are ``huggingface/transformers`` models, so they need to be used with the ``--model_name_or_path`` option. They can also be loaded directly in ``huggingface/transformers`` using ``LEDForConditionalGeneration.from_pretrained()``.

The Google Drive folder containing my contributions to the below models is available `at this link <https://drive.google.com/drive/folders/18W_wJ5ovpSN98AT4626JtEJUCW_wA4ax>`__.

+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| Name (Shortcut Code)                                                                              | Initialized From                                                                        | GDrive Download                                                                            |
+===================================================================================================+=========================================================================================+============================================================================================+
| `allenai/led-base-16384 <https://huggingface.co/allenai/led-base-16384>`_                         | `facebook/bart-base <https://huggingface.co/facebook/bart-large>`_                      |                                                                                            |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `allenai/led-large-16384 <https://huggingface.co/allenai/led-large-16384>`_                       | `facebook/bart-large <https://huggingface.co/facebook/bart-large>`_                     |                                                                                            |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+
| `HHousen/distil-led-large-cnn-16384 <https://huggingface.co/HHousen/distil-led-large-cnn-16384>`_ | `sshleifer/distilbart-cnn-12-6 <https://huggingface.co/sshleifer/distilbart-cnn-12-6>`_ | `Folder Link <https://drive.google.com/drive/folders/1BTM0ccphhHrKt7A1c87DwXU4KbnPn_er>`__ |
+---------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------+

.. note:: In pervious versions of TransformerSum, this section listed models that could be used with the outdated LED model (using custom versions of ``huggingface/transformers`` and ``allenai/longformer``). Those models can still be found in `this Google Drive Folder <https://drive.google.com/drive/folders/1Nwy9_ZNlBpLrgaE8SLaw7BhiyjzY7Coo>`__.


arXiv-PubMed
------------

+------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| Name                         | Comments | Model Download                                                                                              | Data Download |
+==============================+==========+=============================================================================================================+===============+
| led-base-4096-arxiv-pubmed   | None     | `Model <https://drive.google.com/uc?id=>`__ & `All Checkpoints <https://drive.google.com/drive/folders/>`__ | Not yet..     |
+------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| led-large-4096-arxiv-pubmed  | None     | Not yet...                                                                                                  | Not yet..     |
+------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| led-base-16384-arxiv-pubmed  | None     | Not yet...                                                                                                  | Not yet..     |
+------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+
| led-large-16384-arxiv-pubmed | None     | Not yet...                                                                                                  | Not yet..     |
+------------------------------+----------+-------------------------------------------------------------------------------------------------------------+---------------+

arXiv-PubMed ROUGE Scores
^^^^^^^^^^^^^^^^^^^^^^^^^

Test set results on the arXiv-PubMed dataset using ROUGE F\ :sub:`1`\ .

+------------------------------+------------+------------+------------+-------------+
| Name                         | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-L-Sum |
+==============================+============+============+============+=============+
| led-base-4096-arxiv-pubmed   | Not yet... | Not yet... | Not yet... | Not yet...  |
+------------------------------+------------+------------+------------+-------------+
| led-large-4096-arxiv-pubmed  | Not yet... | Not yet... | Not yet... | Not yet...  |
+------------------------------+------------+------------+------------+-------------+
| led-base-16384-arxiv-pubmed  | Not yet... | Not yet... | Not yet... | Not yet...  |
+------------------------------+------------+------------+------------+-------------+
| led-large-16384-arxiv-pubmed | Not yet... | Not yet... | Not yet... | Not yet...  |
+------------------------------+------------+------------+------------+-------------+

Individual ArXiv and PubMed models
----------------------------------

The huggingface model hub has two pre-trained models for long text summarization: `allenai/led-large-16384-arxiv <https://huggingface.co/allenai/led-large-16384-arxiv>`_ and `patrickvonplaten/led-large-16384-pubmed <https://huggingface.co/patrickvonplaten/led-large-16384-pubmed>`_. These models can be used with `pipelines <https://huggingface.co/transformers/main_classes/pipelines.html>`__ to easily summarize long documents. Please see their model cards (by clicking on their names above) for more information.
