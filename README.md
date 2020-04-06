# TransformerExtSum
> A model to perform neural extractive summarization using machine learning transformer models and a tool to convert abstractive summarization datasets to the extractive task.

**Attributions:**

* Code heavily inspired by the following projects:
  * Adapting BERT for Extractive Summariation: [BertSum](https://github.com/nlpyang/BertSum)
  * Word/Sentence Embeddings: [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
  * CNN/CM Dataset: [cnn-dailymail](https://github.com/artmatsak/cnn-dailymail)
  * PyTorch Lightning Classifier: [lightning-text-classification](https://github.com/ricardorei/lightning-text-classification)
* Important projects utilized:
  * PyTorch: [pytorch](https://github.com/pytorch/pytorch/)
  * Training code: [pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning/)
  * Transformer Models: [huggingface/transformers](https://github.com/huggingface/transformers)

## Pre-trained Models

None yet. Please wait.

## Install

Installation is made easy due to conda environments. Simply run this command from the root project directory: `conda env create --file environment.yml` and conda will create and environment called `transformerextsum` with all the required packages from [environment.yml](environment.yml). The spacy `en_core_web_sm` model is required for the [convert_to_extractive.py](convert_to_extractive.py) script to detect sentence boundaries.

### Step-by-Step Instructions

1. Clone this repository: `git clone https://github.com/HHousen/transformerextsum.git`.
2. Change to project directory: `cd transformerextsum`.
3. Run installation command: `conda env create --file environment.yml`.
4. **(Optional)** If using the [convert_to_extractive.py](convert_to_extractive.py) script then download the `en_core_web_sm` spacy model: `python -m spacy download en_core_web_sm`.

## Supported Datasets

Currently only the CNN/DM summarization dataset is supported. The original processing code is available at [abisee/cnn-dailymail](https://github.com/abisee/cnn-dailymail), but for this project the [artmatsak/cnn-dailymail](https://github.com/artmatsak/cnn-dailymail) processing code is used since it does not tokenize and writes the data to text file `train.source`, `train.target`, `val.source`, `val.target`, `test.source` and `test.target`, which is the format expected by [convert_to_extractive.py](convert_to_extractive.py). 

### CNN/DM

Download and unzip the stories directories from [here](https://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. The files can be downloaded from the terminal with `gdown`, which can be installed with `pip install gdown`.

```bash
pip install gdown
gdown https://drive.google.com/uc?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ
gdown https://drive.google.com/uc?id=0BwmD_VLjROrfM1BxdkxVaTY2bWs
tar zxf cnn_stories.tgz
tar zxf dailymail_stories.tgz
```

**Note:** The above Google Drive links may be outdated depending on the time you are reading this. Check the [CNN/DM official website](https://cs.nyu.edu/~kcho/DMQA/) for the most up-to-date download links.

Next, run the processing code in the git submodule for [artmatsak/cnn-dailymail](https://github.com/artmatsak/cnn-dailymail) located in `cnn_dailymail_processor`. Run `python make_datafiles.py /path/to/cnn/stories /path/to/dailymail/stories`, replacing `/path/to/cnn/stories` with the path to where you saved the `cnn/stories` directory that you downloaded; similarly for `dailymail/stories`.

For each of the URL lists (`all_train.txt`, `all_val.txt` and `all_test.txt`) in `cnn_dailymail_processor/url_lists`, the corresponding stories are read from file and written to text files `train.source`, `train.target`, `val.source`, `val.target`, and `test.source` and `test.target`. These will be placed in the newly created `cnn_dm` directory.

## Convert Abstractive to Extractive Dataset

Simply run [convert_to_extractive.py](convert_to_extractive.py) with the path to the data. For example, with the CNN/DM dataset downloaded above: `python convert_to_extractive.py ./cnn_dailymail_processor/cnn_dm`. However, the recommended command is: `python convert_to_extractive.py ./cnn_dailymail_processor/cnn_dm --shard_interval 5000 --compression`, the `--shard_interval` processes the file in chunks of `5000` and writes results to disk in chunks of `5000` (saves RAM) and the `--compression` compresses each output chunk with gzip (depending on the dataset reduces space usage requirement by about 1/2 to 1/3). The default output directory is the input directory that was specified, but the output directory can be changed with `--base_output_path` if desired.

Output of `python convert_to_extractive.py --help`:

```bash
usage: convert_to_extractive.py [-h] [--base_output_path BASE_OUTPUT_PATH]
                                [--split_names {train,val,test} [{train,val,test} ...]]
                                [--source_ext SOURCE_EXT]
                                [--target_ext TARGET_EXT]
                                [--oracle_mode {greedy,combination}]
                                [--shard_interval SHARD_INTERVAL]
                                [--n_process N_PROCESS]
                                [--batch_size BATCH_SIZE] [--compression]
                                [--resume]
                                [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                                DIR

Convert an Abstractive Summarization Dataset to the Extractive Task

positional arguments:
  DIR                   path to data directory

optional arguments:
  -h, --help            show this help message and exit
  --base_output_path BASE_OUTPUT_PATH
                        path to output processed data (default is `base_path`)
  --split_names {train,val,test} [{train,val,test} ...]
                        which splits of dataset to process
  --source_ext SOURCE_EXT
                        extension of source files
  --target_ext TARGET_EXT
                        extension of target files
  --oracle_mode {greedy,combination}
                        method to convert abstractive summaries to extractive
                        summaries
  --shard_interval SHARD_INTERVAL
                        how many examples to include in each shard of the
                        dataset (default: no shards)
  --n_process N_PROCESS
                        number of processes for multithreading
  --batch_size BATCH_SIZE
                        number of batches for tokenization
  --compression         use gzip compression when saving data
  --resume              resume from last shard
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: 'Info').
```

## Training an Extractive Summarization Model

Once the dataset has been converted to the extractive task, it can be used as input to a SentencesProcessor, which has a `add_examples()` function too add sets of `(example, labels)` and a `get_features()` function that returns a TensorDataset of extracted features (`input_ids`, `attention_masks`, `labels`, `token_type_ids`, `sent_rep_token_ids`, `sent_rep_token_ids_masks`). Feature extraction runs in parallel and tokenizes text using the tokenizer appropriate for the model specified with `--model_name_or_path`. The tokenizer can be changed to another huggingface/transformers tokenizer with the `--tokenizer_name` option. 

The actual ExtractiveSummarizer LightningModule (which is similar to an nn.Module but with a built-in training loop, more info at the [pytorch_lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/)) implements a `prepare_data()` function, which loads each json file outputted by the [convert_to_extractive.py](convert_to_extractive.py) script,  adds it to a `SentencesProcessor` object (there is one `SentencesProcessor` object for each split of the dataset), and runs `get_features()` once all the exampels have been added. This `prepare_data()` function is automatically called by `pytorch_lightning` and it runs in parallel to call `json_to_dataset()`, which is the function that actually adds the loaded examples. 

Data processed into TensorDatasets by a SentencesProcessor will automatically be saved to the path specified by `--data_path`. If the training command is run again, then instead of running `get_features()`, the model will load it from file using `torch.load()`. The data is only saved after `get_features()` finishes running. A single save file is created for each split of the dataset.

Continuing with the CNN/CM dataset, to train a model for 50,000 steps on the data run: `python main.py --data_path ./cnn_dailymail_processor/cnn_dm --default_save_path ./trained_models --max_steps 50000`.

The `--data_path` argument specifies where the extractive dataset json file are located.
The `--default_save_path` argument specifies where the logs and model weights should be stored.
If you prefer to measure training progress by epochs instead of steps, the `--max_epochs` and `--min_epochs` options exist just for you.

The batch sizes can be changed with the `--train_batch_size`, `--val_batch_size`, and `--test_batch_size` options. 

If the extractive dataset json files are compressed using json, then they will be automatically decompressed during the data preprocessing step of training. 

Output of `python main.py --help`:

```bash
usage: main.py [-h] --default_save_path DEFAULT_SAVE_PATH
               [--learning_rate LEARNING_RATE] [--min_epochs MIN_EPOCHS]
               [--max_epochs MAX_EPOCHS] [--min_steps MIN_STEPS]
               [--max_steps MAX_STEPS]
               [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
               [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH]
               [--gpus GPUS] [--gradient_clip_val GRADIENT_CLIP_VAL]
               [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--model_name_or_path MODEL_NAME_OR_PATH]
               [--model_type MODEL_TYPE] [--tokenizer_name TOKENIZER_NAME]
               [--tokenizer_lowercase] [--max_seq_length MAX_SEQ_LENGTH]
               [--oracle_mode {none,greedy,combination}] --data_path DATA_PATH
               [--num_threads NUM_THREADS]
               [--processing_num_threads PROCESSING_NUM_THREADS]
               [--weight_decay WEIGHT_DECAY] [--adam_epsilon ADAM_EPSILON]
               [--warmup_steps WARMUP_STEPS]
               [--train_batch_size TRAIN_BATCH_SIZE]
               [--val_batch_size VAL_BATCH_SIZE]
               [--test_batch_size TEST_BATCH_SIZE]
               [--processor_no_bert_compatible_cls]
               [--create_token_type_ids {binary,sequential}]
               [--processors {train,valid,test} [{train,valid,test} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --default_save_path DEFAULT_SAVE_PATH
                        Default path for logs and weights
  --learning_rate LEARNING_RATE
                        The initial learning rate for Adam.
  --min_epochs MIN_EPOCHS
                        Limits training to a minimum number of epochs
  --max_epochs MAX_EPOCHS
                        Limits training to a max number number of epochs
  --min_steps MIN_STEPS
                        Limits training to a minimum number number of steps
  --max_steps MAX_STEPS
                        Limits training to a max number number of steps
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates grads every k batches or as set up in the
                        dict.
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Check val every n train epochs.
  --gpus GPUS           Number of GPUs to train on or Which GPUs to train on.
                        (default: -1 (all gpus))
  --gradient_clip_val GRADIENT_CLIP_VAL
                        Gradient clipping value
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: 'Info').
  --model_name_or_path MODEL_NAME_OR_PATH
  --model_type MODEL_TYPE
  --tokenizer_name TOKENIZER_NAME
  --tokenizer_lowercase
  --max_seq_length MAX_SEQ_LENGTH
  --oracle_mode {none,greedy,combination}
  --data_path DATA_PATH
  --num_threads NUM_THREADS
  --processing_num_threads PROCESSING_NUM_THREADS
  --weight_decay WEIGHT_DECAY
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps.
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size per GPU/CPU for training.
  --val_batch_size VAL_BATCH_SIZE
                        Batch size per GPU/CPU for evaluation.
  --test_batch_size TEST_BATCH_SIZE
                        Batch size per GPU/CPU for testing.
  --processor_no_bert_compatible_cls
                        If model uses bert compatible [CLS] tokens for
                        sentence representations.
  --create_token_type_ids {binary,sequential}
                        Create token type ids.
  --processors {train,valid,test} [{train,valid,test} ...]
                        which dataset splits to process
```

All training arguments can be found in the [pytorch_lightning trainer documentation](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html).

## Meta

Hayden Housen – [haydenhousen.com](https://haydenhousen.com)

Distributed under the MIT license. See the [LICENSE](LICENSE) for more information.

<https://github.com/HHousen>

## Contributing

1. Fork it (<https://github.com/HHousen/ai-respiratory-doctor/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
