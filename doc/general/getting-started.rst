Getting Started
===============

Install
-------

Installation is made easy due to conda environments. Simply run this command from the root project directory: ``conda env create --file environment.yml`` and conda will create and environment called ``transformersum`` with all the required packages from ``environment.yml``. The spacy ``en_core_web_sm`` model is required for the ``convert_to_extractive.py`` script to detect sentence boundaries.

Step-by-Step Instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone this repository: ``git clone https://github.com/HHousen/transformersum.git``.
2. Change to project directory: ``cd transformersum``.
3. Run installation command: ``conda env create --file environment.yml``.
4. **(Optional)** If using the ``convert_to_extractive.py`` script then download the ``en_core_web_sm`` spacy model: ``python -m spacy download en_core_web_sm``.