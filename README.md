![TransformerSum Logo](doc/_static/logo.png)

# TransformerSum
> Models to perform neural summarization (extractive and abstractive) using machine learning transformers and a tool to convert abstractive summarization datasets to the extractive task.

[![GitHub license](https://img.shields.io/github/license/HHousen/TransformerSum.svg)](https://github.com/HHousen/TransformerSum/blob/master/LICENSE) [![Github commits](https://img.shields.io/github/last-commit/HHousen/TransformerSum.svg)](https://github.com/HHousen/TransformerSum/commits/master) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Documentation Status](https://readthedocs.org/projects/TransformerSum/badge/?version=latest)](http://TransformerSum.readthedocs.io/?badge=latest) [![GitHub issues](https://img.shields.io/github/issues/HHousen/TransformerSum.svg)](https://GitHub.com/HHousen/TransformerSum/issues/) [![GitHub pull-requests](https://img.shields.io/github/issues-pr/HHousen/TransformerSum.svg)](https://GitHub.com/HHousen/TransformerSum/pull/)

`TransformerSum` is a library that aims to make to easy to *train*, *evaluate*, and *use* machine learning **transformer models** that perform **automatic summarization**. It features tight integration with [huggingface/transformers](https://github.com/huggingface/transformers) which enables the easy usage of a **wide variety of architectures** and **pre-trained models**. There is a heavy emphasis on code **readability** and **interpretability** so that both beginners and experts can build new components. Both the extractive and abstractive model classes are written using [pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning), which handles the PyTorch training loop logic, enabling **easy usage of advanced features** such as 16-bit precision, multi-GPU training, and [much more](https://pytorch-lightning.readthedocs.io/). `TransformerSum` supports both the extractive and abstractive summarization of **long sequences** (4,096 to 16,000 tokens) using the [longformer](https://huggingface.co/transformers/model_doc/longformer.html) (extractive) and [longbart](https://github.com/patil-suraj/longbart) (abstractive), which is a combination of [BART](https://huggingface.co/transformers/model_doc/bart.html) ([paper](https://arxiv.org/abs/1910.13461)) and the longformer. Models are automatically evaluated with the **ROUGE metric** but human tests can be conducted by the user.

Check out [the documentation](https://transformersum.readthedocs.io/en/latest) for usage details.

## Features

* For extractive summarization, compatible with every [huggingface/transformers](https://github.com/huggingface/transformers) transformer encoder model.
* For abstractive summarization, compatible with every [huggingface/transformers](https://github.com/huggingface/transformers) EncoderDecoder model.
* Currently, 10+ pre-trained extractive models available to summarize text trained on 3 datasets (CNN-DM, WikiHow, and ArXiv-PebMed).

* Contains pre-trained models that excel at summarization on resource-limited devices: On CNN-DM, ``mobilebert-uncased-ext-sum`` achieves about 97% of the performance of [BertSum](https://arxiv.org/abs/1903.10318) while containing 4.45 times fewer parameters. It achieves about 94% of the performance of [MatchSum (Zhong et al., 2020)](https://arxiv.org/abs/2004.08795), the current extractive state-of-the-art.
* Contains code to train models that excel at summarizing long sequences: The [longformer](https://huggingface.co/transformers/model_doc/longformer.html) (extractive) and [longbart](https://github.com/patil-suraj/longbart) (abstractive) can summarize sequences of lengths up to 4,096 tokens by default, but can be trained to summarize sequences of more than 16k.

* Integration with [huggingface/nlp](https://github.com/huggingface/nlp) means any summarization dataset in the `nlp` library can be used for both abstractive and extractive training.
* "Smart batching" support to not perform unnecessary calculations (speeds up training).
* Use of `pytorch_lighting` for code readability.
* Extensive documentation.
* Two pooling modes (convert word vectors to sentence embeddings): mean of word embeddings or use the CLS token.

## Pre-trained Models

All pre-trained models (including larger models and other architectures) are located in [the documentation](https://transformersum.readthedocs.io/en/latest). The below is a fraction of the available models.

### Extractive

| Name | Dataset | Comments | R1/R2/RL/RL-Sum | Model Download | Data Download |
|-|-|-|-|-|-|
| mobilebert-uncased-ext-sum | CNN/DM | None | 42.01/19.31/26.89/38.53 | [Model](https://drive.google.com/uc?id=1-4MTKOXp1hkPJ_pK6yOCDULC0JVly3SE) | [CNN/DM Bert Uncased](https://drive.google.com/uc?id=100ZE4fVU73EU3K_EGktrYDoMSLJ6EUQW) |
| distilroberta-base-ext-sum | CNN/DM | None | 42.87/20.02/27.46/39.31 | [Model](https://drive.google.com/uc?id=1-2TZe28K8inHoJr2-WuVivj2qwBn7tFs) | [CNN/DM Roberta](https://drive.google.com/uc?id=1-L7UOYe69dD--OPGCa4sS0QQEnZNb_Vb) |
| roberta-base-ext-sum | CNN/DM | None | 43.24/20.36/27.64/39.65 | [Model](https://drive.google.com/uc?id=18ZlImBv1P7VmDPUpiQHF9frk-q3AFfD0) | [CNN/DM Roberta](https://drive.google.com/uc?id=1-L7UOYe69dD--OPGCa4sS0QQEnZNb_Vb) |
| mobilebert-uncased-ext-sum | WikiHow | None | 30.72/8.78/19.18/28.59 | [Model](https://drive.google.com/uc?id=1-H7kuojMW50gVnC5flEjQkbHRp-WQtaF) | [WikiHow Bert Uncased](https://drive.google.com/uc?id=1-IO2AgjDsJcbrmsM3R4UIRM2bMHR-Dae) |
| distilroberta-base-ext-sum | WikiHow | None | 31.07/8.96/19.34/28.95 | [Model](https://drive.google.com/uc?id=1-3NV3TdRcTta9JTi9Kh0sWtoNLEdWrY1) | [WikiHow Roberta](https://drive.google.com/uc?id=1-aQMjCEQlKhEcimMW_WJwQusNScIT2Uf) |
| roberta-base-ext-sum | WikiHow | None | Not yet... | Not yet... | [WikiHow Roberta](https://drive.google.com/uc?id=1-aQMjCEQlKhEcimMW_WJwQusNScIT2Uf) |
| mobilebert-uncased-ext-sum | arXiv-PubMed | None | 27.52/7.26/15.53/24.37 | [Model](https://drive.google.com/uc?id=1-KoEYNC2ZiRoP97V4tjXFy-YfltNBape) | [arXiv-PubMed Bert Uncased](https://drive.google.com/uc?id=1-GbxiYkXkK7qcde37JtKtH5U7iIpdrnI) |
| distilroberta-base-ext-sum | arXiv-PubMed | None | 34.70/12.16/19.52/30.82 | [Model](https://drive.google.com/uc?id=1-8xVR72-jWtIxvl6DYvcND2yVc0gxjGR) | [arXiv-PubMed Roberta](https://drive.google.com/uc?id=11pVkVO1ivC3okWq-l_xW1qQmagDE5Htt) |
| roberta-base-ext-sum | arXiv-PubMed | None | Not yet... | Not yet... | [arXiv-PubMed Roberta](https://drive.google.com/uc?id=11pVkVO1ivC3okWq-l_xW1qQmagDE5Htt) |

### Abstractive

| Name | Dataset | Comments | Model Download |
|-|-|-|-|
| longbart-base-4096-abs-sum | arXiv-PubMed | None | Not yet... |

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
