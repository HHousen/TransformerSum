# TransformerSum
> Models to perform neural summarization (extractive and abstractive) using machine learning transformers and a tool to convert abstractive summarization datasets to the extractive task.

[![GitHub license](https://img.shields.io/github/license/HHousen/TransformerSum.svg)](https://github.com/HHousen/TransformerSum/blob/master/LICENSE) [![Github commits](https://img.shields.io/github/last-commit/HHousen/TransformerSum.svg)](https://github.com/HHousen/TransformerSum/commits/master) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Documentation Status](https://readthedocs.org/projects/TransformerSum/badge/?version=latest)](http://TransformerSum.readthedocs.io/?badge=latest) [![GitHub issues](https://img.shields.io/github/issues/HHousen/TransformerSum.svg)](https://GitHub.com/HHousen/TransformerSum/issues/) [![GitHub pull-requests](https://img.shields.io/github/issues-pr/HHousen/TransformerSum.svg)](https://GitHub.com/HHousen/TransformerSum/pull/)

Check out [the documentation](https://transformersum.readthedocs.io/en/latest) for usage details.

## Pre-trained Models

All pre-trained models (including larger models and other architectures) are located in `the documentation <https://transformersum.readthedocs.io/en/latest>`_. The below is just a small fraction of the available models.

### Extractive

| Name | Dataset | Comments | Model Download | Data Download |
|-|-|-|-|-|
| distilroberta-base-ext-sum | CNN/DM | None | Not yet... | [CNN/DM Roberta](https://drive.google.com/uc?id=1g2bAAUuDvt3WZ3giSDaN7j4xew-RGfT3) |
| roberta-base-ext-sum | CNN/DM | None | Not yet... | [CNN/DM Roberta](https://drive.google.com/uc?id=1g2bAAUuDvt3WZ3giSDaN7j4xew-RGfT3) |
| distilroberta-base-ext-sum | WikiHow | None | Not yet... | [WikiHow Roberta]() |
| roberta-base-ext-sum | WikiHow | None | Not yet... | [WikiHow Roberta]() |
| distilroberta-base-ext-sum | arXiv-PubMed | None | Not yet... | [arXiv-PubMed Roberta]() |
| roberta-base-ext-sum | arXiv-PubMed | None | Not yet... | [arXiv-PubMed Roberta]() |

### Abstractive

| Name | Dataset | Comments | Model Download |
|-|-|-|-|
| distilroberta-base-abs-sum | CNN/DM | None | Not yet... |
| roberta-base-abs-sum | CNN/DM | None | Not yet... |
| distilroberta-base-abs-sum | WikiHow | None | Not yet... |
| roberta-base-abs-sum | WikiHow | None | Not yet... |
| distilroberta-base-abs-sum | arXiv-PubMed | None | Not yet... |
| roberta-base-abs-sum | arXiv-PubMed | None | Not yet... |

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
