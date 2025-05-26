![TransformerSum Logo](doc/_static/logo.png)

# TransformerSum
> Models to perform neural summarization (extractive and abstractive) using machine learning transformers and a tool to convert abstractive summarization datasets to the extractive task.

[![GitHub license](https://img.shields.io/github/license/HHousen/TransformerSum.svg)](https://github.com/HHousen/TransformerSum/blob/master/LICENSE) [![Github commits](https://img.shields.io/github/last-commit/HHousen/TransformerSum.svg)](https://github.com/HHousen/TransformerSum/commits/master) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Documentation Status](https://readthedocs.org/projects/transformersum/badge/?version=latest)](https://transformersum.readthedocs.io/en/latest/?badge=latest) [![GitHub issues](https://img.shields.io/github/issues/HHousen/TransformerSum.svg)](https://GitHub.com/HHousen/TransformerSum/issues/) [![GitHub pull-requests](https://img.shields.io/github/issues-pr/HHousen/TransformerSum.svg)](https://GitHub.com/HHousen/TransformerSum/pull/) [![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/HHousen/TransformerSum/?ref=repository-badge)

`TransformerSum` is a library that aims to make it easy to *train*, *evaluate*, and *use* machine learning **transformer models** that perform **automatic summarization**. It features tight integration with [huggingface/transformers](https://github.com/huggingface/transformers) which enables the easy usage of a **wide variety of architectures** and **pre-trained models**. There is a heavy emphasis on code **readability** and **interpretability** so that both beginners and experts can build new components. Both the extractive and abstractive model classes are written using [pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning), which handles the PyTorch training loop logic, enabling **easy usage of advanced features** such as 16-bit precision, multi-GPU training, and [much more](https://pytorch-lightning.readthedocs.io/). `TransformerSum` supports both the extractive and abstractive summarization of **long sequences** (4,096 to 16,384 tokens) using the [longformer](https://huggingface.co/transformers/model_doc/longformer.html) (extractive) and [LongformerEncoderDecoder](https://github.com/allenai/longformer/tree/encoderdecoder) (abstractive), which is a combination of [BART](https://huggingface.co/transformers/model_doc/bart.html) ([paper](https://arxiv.org/abs/1910.13461)) and the longformer. `TransformerSum` also contains models that can run on resource-limited devices while still maintaining high levels of accuracy. Models are automatically evaluated with the **ROUGE metric** but human tests can be conducted by the user.

**Check out [the documentation](https://transformersum.readthedocs.io/en/latest) for usage details.**

## Features

* For extractive summarization, compatible with every [huggingface/transformers](https://github.com/huggingface/transformers) transformer encoder model.
* For abstractive summarization, compatible with every [huggingface/transformers](https://github.com/huggingface/transformers) EncoderDecoder and Seq2Seq model.
* Currently, 10+ pre-trained extractive models available to summarize text trained on 3 datasets (CNN-DM, WikiHow, and ArXiv-PebMed).

* Contains pre-trained models that excel at summarization on resource-limited devices: On CNN-DM, ``mobilebert-uncased-ext-sum`` achieves about 97% of the performance of [BertSum](https://arxiv.org/abs/1903.10318) while containing 4.45 times fewer parameters. It achieves about 94% of the performance of [MatchSum (Zhong et al., 2020)](https://arxiv.org/abs/2004.08795), the current extractive state-of-the-art.
* Contains code to train models that excel at summarizing long sequences: The [longformer](https://huggingface.co/transformers/model_doc/longformer.html) (extractive) and [LongformerEncoderDecoder](https://github.com/allenai/longformer/tree/encoderdecoder) (abstractive) can summarize sequences of lengths up to 4,096 tokens by default, but can be trained to summarize sequences of more than 16k tokens.

* Integration with [huggingface/nlp](https://github.com/huggingface/nlp) means any summarization dataset in the `nlp` library can be used for both abstractive and extractive training.
* "Smart batching" (extractive) and trimming (abstractive) support to not perform unnecessary calculations (speeds up training).
* Use of `pytorch_lightning` for code readability.
* Extensive documentation.
* Three pooling modes (convert word vectors to sentence embeddings): mean or max of word embeddings in addition to the CLS token.

## Pre-trained Models

All pre-trained models (including larger models and other architectures) are located in [the documentation](https://transformersum.readthedocs.io/en/latest). The below is a fraction of the available models.

### Extractive

| Name | Dataset | Comments | R1/R2/RL/RL-Sum | Model Download | Data Download |
|-|-|-|-|-|-|
| mobilebert-uncased-ext-sum | CNN/DM | None | 42.01/19.31/26.89/38.53 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/models/mobilebert-uncased_8e-5/epoch=3.ckpt) | [CNN/DM Bert Uncased](https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz) |
| distilroberta-base-ext-sum | CNN/DM | None | 42.87/20.02/27.46/39.31 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/models/distilroberta-base/epoch=3.ckpt) | [CNN/DM Roberta](https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/roberta-base/roberta-base.tar.gz) |
| roberta-base-ext-sum | CNN/DM | None | 43.24/20.36/27.64/39.65 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/models/roberta-base/epoch=3.ckpt) | [CNN/DM Roberta](https://huggingface.co/HHousen/TransformerSum/blob/main/CNN-DM/cnn_dm_pt_lists_5000/roberta-base/roberta-base.tar.gz) |
| mobilebert-uncased-ext-sum | WikiHow | None | 30.72/8.78/19.18/28.59 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/models/mobilebert-uncased/epoch=3.ckpt) | [WikiHow Bert Uncased](https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz) |
| distilroberta-base-ext-sum | WikiHow | None | 31.07/8.96/19.34/28.95 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/models/distilroberta-base/epoch=3.ckpt) | [WikiHow Roberta](https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/roberta-base/roberta-base.tar.gz) |
| roberta-base-ext-sum | WikiHow | None | 31.26/09.09/19.47/29.14 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/models/roberta-base/epoch=2.ckpt) | [WikiHow Roberta](https://huggingface.co/HHousen/TransformerSum/blob/main/WikiHow/wikihow_pt_lists_5000/roberta-base/roberta-base.tar.gz) |
| mobilebert-uncased-ext-sum | arXiv-PubMed | None | 33.97/11.74/19.63/30.19 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/models/mobilebert-uncased/epoch=3.ckpt) | [arXiv-PubMed Bert Uncased](https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/bert-base-uncased/bert-base-uncased.tar.gz) |
| distilroberta-base-ext-sum | arXiv-PubMed | None | 34.70/12.16/19.52/30.82 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/models/distilroberta-base/epoch=3.ckpt) | [arXiv-PubMed Roberta](https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/roberta-base/roberta-base.tar.gz) |
| roberta-base-ext-sum | arXiv-PubMed | None | 34.81/12.26/19.65/30.91 | [Model](https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/models/roberta-base/epoch=2.ckpt) | [arXiv-PubMed Roberta](https://huggingface.co/HHousen/TransformerSum/blob/main/arXiv-PubMed/arxiv-pubmed_pt_lists_5000/roberta-base/roberta-base.tar.gz) |

### Abstractive

| Name | Dataset | Comments | Model Download |
|-|-|-|-|
| longformer-encdec-8192-bart-large-abs-sum | arXiv-PubMed | None | Not yet... |

## Install

Installation is made easy due to conda environments. Simply run this command from the root project directory: `conda env create --file environment.yml` and conda will create and environment called `transformersum` with all the required packages from [environment.yml](environment.yml). The spacy `en_core_web_sm` model is required for the [convert_to_extractive.py](convert_to_extractive.py) script to detect sentence boundaries.

### Step-by-Step Instructions

1. Clone this repository: `git clone https://github.com/HHousen/transformersum.git`.
2. Change to project directory: `cd transformersum`.
3. Run installation command: `conda env create --file environment.yml`.
4. **(Optional)** If using the [convert_to_extractive.py](convert_to_extractive.py) script then download the `en_core_web_sm` spacy model: `python -m spacy download en_core_web_sm`.

## Meta

[![ForTheBadge built-with-love](https://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/HHousen/)

Hayden Housen – [haydenhousen.com](https://haydenhousen.com)

Distributed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) for more information.

<https://github.com/HHousen>

### Attributions

* Code heavily inspired by the following projects:
  * Adapting BERT for Extractive Summariation: [BertSum](https://github.com/nlpyang/BertSum)
  * Text Summarization with Pretrained Encoders: [PreSumm](https://github.com/nlpyang/PreSumm)
  * Word/Sentence Embeddings: [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
  * CNN/CM Dataset: [cnn-dailymail](https://github.com/artmatsak/cnn-dailymail)
  * PyTorch Lightning Classifier: [lightning-text-classification](https://github.com/ricardorei/lightning-text-classification)
* Important projects utilized:
  * PyTorch: [pytorch](https://github.com/pytorch/pytorch/)
  * Training code: [pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning/)
  * Transformer Models: [huggingface/transformers](https://github.com/huggingface/transformers)

## Contributing

All Pull Requests are greatly welcomed.

Questions? Commends? Issues? Don't hesitate to open an [issue](https://github.com/HHousen/TransformerSum/issues/new) and briefly describe what you are experiencing (with any error logs if necessary). Thanks.

1. Fork it (<https://github.com/HHousen/TransformerSum/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
