# Experiments

Interactive charts, graphs, raw data, run commands, hyperparameter choices, and more for all experiments are publicly available on the [TransformerExtSum Weights & Biases page](https://app.wandb.ai/hhousen/transformerextsum).

**Reproducibility Notes:**

If you are unable to reproduce the results for the experiments below by following the instructions for each experiment, then please open an [issue](https://github.com/HHousen/TransformerExtSum/issues/new). The following is a list of things to double check if you cannot reproduce the results:

* If you are using `--overfit_pct`, then `overfit_pct` percent of the testing data is being used as well as `overfit_pct` percent of the training data. Due to the way `pytorch_lightning` was written, it is necessary to use the same `batch_size` when using `overfit_pct` in order to get the exact same results. I currently am not sure why this is the case but removing `overfit_pct` and using different `batch_size`s produces identical results. Open an [issue](https://github.com/HHousen/TransformerExtSum/issues/new) or submit a pull request if you know why.
* Have another note that should be stated here? Open an [issue](https://github.com/HHousen/TransformerExtSum/issues/new). All contributions are very helpful.

## Loss Functions

The loss function implementation can be found in [model.py](model.py) with signature `compute_loss(self, outputs, labels, mask)`. The function uses `nn.BCELoss` with `reduction="none"` and then applies 5 different reduction techniques. Special reduction methods were needed to ignore padding and operate on the multi-class-per-document approach (each input is assigned more than one of the same class) that this research uses to perform extractive summarization. See the comments throughout the function for more information. The five different reduction methods were tested with the `distilbert-base-uncased` word embedding model and the `pooling_mode` set to `sent_rep_tokens`. Training time is just under 4 hours on a Tesla P100 (3h52m average).

The `--loss_key` argument specifies the reduction method to use. It can be one of the following: `loss_total`, `loss_total_norm_batch`, `loss_avg_seq_sum`, `loss_avg_seq_mean`, `loss_avg`.

Full command used to run the tests:

```
python main.py \
--model_name_or_path distilbert-base-uncased \
--no_use_token_type_ids \
--pooling_mode sent_rep_tokens \
--data_path ./cnn_dm_pt/bert-base-uncased \
--max_epochs 3 \
--accumulate_grad_batches 2 \
--warmup_steps 1800 \
--overfit_pct 0.6 \
--gradient_clip_val 1.0 \
--optimizer_type adamw \
--use_scheduler linear \
--profiler \
--do_train --do_test \
--loss_key [Loss Key Here] \
--batch_size 32
```

### Loss Functions Results

Graph Legend Description: The `loss-test` label (the first part) is the experiment, which indicates the loss reduction method that was tested. The second part of each key is the graphed quantity. For example, the first line of the key for the first graph in the `Outliers Included` section below indicates that `loss_avg` was tested and that its results as measured by the `loss_avg_seq_mean` reduction method are shown in brown. The train results are solid brown and the validation results are dotted brown.

**Outliers Included:**

<img src="loss_functions/loss_avg_seq_mean_outliers.png" width="450" /> <img src="loss_functions/loss_total_outliers.png" width="450" />

**No Outliers:**

<img src="loss_functions/loss_avg_seq_sum.png" width="450" /> <img src="loss_functions/loss_avg_seq_sum.png" width="450" />

<img src="loss_functions/loss_total_norm_batch.png" width="450" /> <img src="loss_functions/loss_total_norm_batch.png" width="450" />

<img src="loss_functions/loss_total.png" width="450" /> <img src="loss_functions/loss_avg_seq_mean_val_only.png" width="450" />

The CSV files the were used to generate the above graphs can be found in `experiments/loss_functions`.

Based on the results, `loss_avg_seq_mean` was chosen as the default.

## Word Embedding Models

Different transformer models of various architectures and sizes were tested.

Tested Models:

| Model Type | Model Key                                                 | Batch Size |
|------------|-----------------------------------------------------------|------------|
| Distil*    | `distilbert-base-uncased`, `distilroberta-base`           | 16         |
| Base       | `bert-base-uncased`, `roberta-base`, `albert-base-v2`     | 16         |
| Large      | `bert-large-uncased`, `roberta-large`, `albert-xlarge-v2` | 4          |

**Albert Info:** The above batch sizes are true except for `albert` models, which have special batch sizes due to the increased memory needed to train them*. *`albert-base-v2` was trained with a batch size of `12` and `albert-xlarge-v2` with a batch size of `2`.* 

| Model          | Parameters | Layers | Hidden | Heads | Embedding | Parameter-sharing |
|----------------|------------|--------|--------|-------|-----------|-------------------|
| BERT-base      | 110M       | 12     | 768    | 12    | 768       | False             |
| BERT-large     | 340M       | 24     | 1024   | 16    | 1024      | False             |
| ALBERT-base    | 12M        | 12     | 768    | 12    | 128       | True              |
| ALBERT-large   | 18M        | 24     | 1024   | 16    | 128       | True              |
| ALBERT-xlarge  | 59M        | 24     | 2048   | 32    | 128       | True              |
| ALBERT-xxlarge | 233M       | 12     | 4096   | 64    | 128       | True              |

*The huggingface/transformers documentation says "ALBERT uses repeating layers which results in a small memory footprint." This may be true but I found that the normal batch sizes I used for the base and large models would crash the training script when `albert` models were used. Thus, the batch sizes were decreased. The advantage that of `albert` that I found was incredibly small model weight checkpoint files (see results below for sizes).

All models were trained for 3 epochs (except `albert-xlarge-v2`) (which will result in different numbers of steps but will ensure that each model saw the same amount of information), using the AdamW optimizer with a linear scheduler with 1800 steps of warmup. Gradients were accumulated every 2 batches and clipped at 1.0. **Only 60% of the data was used** (to decrease training time, but also will provide similar results if all the data was used). `--no_use_token_type_ids` was set if the model was not compatible with token type ids.

Full command used to run the tests:

```
python main.py \
--model_name_or_path [Model Name] \
--model_type [Model Type] \
--pooling_mode sent_rep_tokens \
--data_path ./cnn_dm_pt/[Model Type]-base \
--max_epochs 3 \
--accumulate_grad_batches 2 \
--warmup_steps 1800 \
--overfit_pct 0.6 \
--gradient_clip_val 1.0 \
--optimizer_type adamw \
--use_scheduler linear \
--profiler \
--do_train --do_test \
--batch_size [Batch Size]
```

### WEB Results

The CSV files the were used to generate the below graphs can be found in `experiments/web`.

All `ROUGE Scores` are test set results on the CNN/DailyMail dataset using ROUGE F<sub>1</sub>.

All model sizes are not compressed. They are the raw `.ckpt` output file sizes of the best performing epoch by `val_loss`.

#### Final (Combined) Results

The `loss_total`, `loss_avg_seq_sum`, and `loss_total_norm_batch` loss reduction techniques depend on the batch size. That is, the larger the batch size, the larger these losses will be. The `loss_avg_seq_mean` and `loss_avg` do not depend on the batch size since they are averages instead of totals. Therefore, only the non-batch-size-dependent metrics were used for the final results because difference batch sizes were used.

#### Distil* Models

More information about distil* models found in the [huggingface/transformers examples](https://github.com/huggingface/transformers/tree/master/examples/distillation).

**Important Note:** Distil* models do not accept token type ids. So set `--no_use_token_type_ids` while training using the above command.

**Training Times and Model Sizes:**

| Model Key                 | Time       | Model Size |
|---------------------------|------------|------------|
| `distilbert-base-uncased` | 4h 5m 30s  | 810.6MB    |
| `distilroberta-base`      | 4h 12m 53s | 995.0MB    |

**ROUGE Scores:**

| Name                    | ROUGE-1    | ROUGE-2    | ROUGE-L    |
|-------------------------|------------|------------|------------|
| distilbert-base-uncased | 40.1       | 18.1       | 26.0       |
| distilroberta-base      | 40.9       | 18.7       | 26.4       |

**Outliers Included:**

<img src="word_embedding_models/distil_loss_avg_seq_mean_outliers.png" width="450" /> <img src="word_embedding_models/distil_loss_total_outliers.png" width="450" />

**No Outliers:**

<img src="word_embedding_models/distil_loss_avg_seq_mean.png" width="450" /> <img src="word_embedding_models/distil_loss_avg_seq_sum.png" width="450" />

<img src="word_embedding_models/distil_loss_total_norm_batch.png" width="450" /> <img src="word_embedding_models/distil_loss_avg.png" width="450" />

<img src="word_embedding_models/distil_loss_total.png" width="450" /> <img src="word_embedding_models/distil_loss_avg_seq_mean_val_only.png" width="450" />

#### Base Models

**Important Note:** `roberta-base` does not accept token type ids. So set `--no_use_token_type_ids` while training using the above command.

**Training Times and Model Sizes:**

| Model Key           | Time       | Model Size |
|---------------------|------------|------------|
| `bert-base-uncased` | 7h 56m 39s | 1.3GB      |
| `roberta-base`      | 7h 52m 0s  | 1.5GB      |
| `albert-base-v2`    | 7h 32m 19s | 149.7MB    |

**ROUGE Scores:**

| Name              | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------------------|---------|---------|---------|
| bert-base-uncased | 40.2    | 18.2    | 26.1    |
| roberta-base      | 42.3    | 20.1    | 27.4    |
| albert-base-v2    | 40.5    | 18.4    | 26.1    |

**Outliers Included:**

<img src="word_embedding_models/base_loss_avg_seq_mean_outliers.png" width="450" /> <img src="word_embedding_models/base_loss_total_outliers.png" width="450" />

**No Outliers:**

<img src="word_embedding_models/base_loss_avg_seq_mean.png" width="450" /> <img src="word_embedding_models/base_loss_avg_seq_sum.png" width="450" />

<img src="word_embedding_models/base_loss_total_norm_batch.png" width="450" /> <img src="word_embedding_models/base_loss_avg.png" width="450" />

<img src="word_embedding_models/base_loss_total.png" width="450" /> <img src="word_embedding_models/base_loss_avg_seq_mean_val_only.png" width="450" />

**Relative Time:**

This is included because the batch size for `albert-base-v2` had to be lowered to 12 (from 16).

<img src="word_embedding_models/base_loss_avg_seq_mean_reltime.png" width="450" />


#### Large Models

**Important Note:** `roberta-large` does not accept token type ids. So set `--no_use_token_type_ids` while training using the above command.

**More Important Note:** `albert-xlarge-v2` (batch size 2) was set to be trained with for 2 epochs instead of 3, but was stopped early at `global_step` 56394.

**Training Times and Model Sizes:**

| Model Key            | Time        | Model Size |
|----------------------|-------------|------------|
| `bert-large-uncased` | 17h 55m 18s | 4.0GB      |
| `roberta-large`      | 18h 32m 28s | 4.3GB      |
| `albert-xlarge-v2`   | 21h 15m 54s | 708.9MB    |

**ROUGE Scores:**

| Name               | ROUGE-1    | ROUGE-2    | ROUGE-L    |
|--------------------|------------|------------|------------|
| bert-large-uncased | 41.5       | 19.3       | 27.0       |
| roberta-large      | 41.5       | 19.3       | 27.0       |
| albert-xlarge-v2   | 40.7       | 18.4       | 26.1       |

**Outliers Included:**

<img src="word_embedding_models/large_loss_avg_seq_mean_outliers.png" width="450" /> <img src="word_embedding_models/large_loss_total_outliers.png" width="450" />

**No Outliers:**

<img src="word_embedding_models/large_loss_avg_seq_mean.png" width="450" /> <img src="word_embedding_models/large_loss_avg_seq_sum.png" width="450" />

<img src="word_embedding_models/large_loss_total_norm_batch.png" width="450" /> <img src="word_embedding_models/large_loss_avg.png" width="450" />

<img src="word_embedding_models/large_loss_total.png" width="450" /> <img src="word_embedding_models/large_loss_avg_seq_mean_val_only.png" width="450" />

**Relative Time:**

This is included because the batch size for `albert-large-v2` had to be lowered to 2 (from 4).

<img src="word_embedding_models/large_loss_avg_seq_mean_reltime.png" width="450" />

## Pooling Mode

See [the main README.md](../README.md) for more information on what the pooling model is.

The two options, `sent_rep_tokens` and `mean_tokens`, were both tested with the `bert-base-uncased` and `distilbert-base-uncased` word embedding models.

Full command used to run the tests:

```
python main.py \
--model_name_or_path [Model Name] \
--model_type [Model Type] \
--pooling_mode [`mean_tokens` or `sent_rep_tokens`] \
--data_path ./cnn_dm_pt/[Model Type]-base \
--max_epochs 3 \
--accumulate_grad_batches 2 \
--warmup_steps 1800 \
--overfit_pct 0.6 \
--gradient_clip_val 1.0 \
--optimizer_type adamw \
--use_scheduler linear \
--profiler \
--do_train --do_test \
--batch_size 16
```

### Pooling Mode Results

**Training Times and Model Sizes:**

| Model Key                                 | Time       | Model Size |
|-------------------------------------------|------------|------------|
| `distilbert-base-uncased` mean_tokens     | 5h 18m 1s  | 810.6MB    |
| `distilbert-base-uncased` sent_rep_tokens | 4h 5m 30s  | 810.6MB    |
| `bert-base-uncased` mean_tokens           | 8h 22m 46s | 1.3GB      |
| `bert-base-uncased` sent_rep_tokens       | 7h 56m 39s | 1.3GB      |

**ROUGE Scores:**

| Name                                    | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-----------------------------------------|---------|---------|---------|
| distilbert-base-uncased mean_tokens     | 41.1    | 18.8    | 26.5    |
| distilbert-base-uncased sent_rep_tokens | 40.1    | 18.1    | 26.0    |
| bert-base-uncased mean_tokens           | 40.7    | 18.7    | 26.6    |
| bert-base-uncased sent_rep_tokens       | 40.2    | 18.2    | 26.1    |

**Main Takeaway:** Using the `mean_tokens` `pooling_mode` is associated with a *0.617 average ROUGE F<sub>1</sub> score improvement* over the `sent_rep_tokens` `pooling_mode`. This improvement is at the cost of a *49.3 average minute (2959 seconds) increase in training time*.

**Outliers Included:**

<img src="pooling_mode/loss_avg_seq_mean_outliers.png" width="450" /> <img src="pooling_mode/loss_total_outliers.png" width="450" />

**No Outliers:**

<img src="pooling_mode/loss_avg_seq_sum.png" width="450" /> <img src="pooling_mode/loss_avg_seq_mean.png" width="450" />

<img src="pooling_mode/loss_total_norm_batch.png" width="450" /> <img src="pooling_mode/loss_avg.png" width="450" />

<img src="pooling_mode/loss_total.png" width="450" /> <img src="pooling_mode/loss_avg_seq_mean_val_only.png" width="450" />

**Relative Time:**

<img src="pooling_mode/loss_avg_seq_mean_reltime.png" width="450" />

## Classifier/Encoder

The classifier/encoder is responsible for removing the hidden features from each sentence embedding and converting them to a single number. The `linear`, `transformer` (with 2 layers), `transformer` (with 6 layers "`--classifier_transformer_num_layers 6`"), and `transformer_linear` options were tested with the `distilbert-base-uncased` model. The `transformer_linear` test has a transformer with *2 layers* (like the `transformer` test).

Unlike the experiments prior to this one (above), the "Classifier/Encoder" experiment used a `--train_percent_check` of 0.6, `--val_percent_check` of 0.6 and **`--test_percent_check` of 1.0**. All of the data was used for testing whereas 60% of it was used for training and validation.

Full command used to run the tests:

```
python main.py \
--model_name_or_path [Model Name] \
--model_type distilbert \
--no_use_token_type_ids \
--classifier [`linear` or `transformer` or `transformer_linear`] \
[--classifier_transformer_num_layers 6 \]
--data_path ./cnn_dm_pt/bert-base-uncased \
--max_epochs 3 \
--accumulate_grad_batches 2 \
--warmup_steps 1800 \
--train_percent_check 0.6 --val_percent_check 0.6 --test_percent_check 1.0 \
--gradient_clip_val 1.0 \
--optimizer_type adamw \
--use_scheduler linear \
--profiler \
--do_train --do_test \
--batch_size 16
```

### Classifier/Encoder Results

**Training Times and Model Sizes:**

| Model Key                | Time       | Model Size |
|--------------------------|------------|------------|
| `linear`                 | 3h 59m 1s  | 810.6MB    |
| `transformer` (2 layers) | 4h 9m 29s  | 928.8MB    |
| `transformer` (6 layers) | 4h 21m 29s | 1.2GB      |
| `transformer_linear`     | 4h 9m 59s  | 943.0MB    |

**ROUGE Scores:**

| Name                     | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------------------------|---------|---------|---------|
| `linear`                 | 41.2    | 18.9    | 26.5    |
| `transformer` (2 layers) | 41.2    | 18.8    | 26.5    |
| `transformer` (6 layers) | 41.0    | 18.9    | 26.5    |
| `transformer_linear`     | 40.9    | 18.7    | 26.6    |

**Main Takeaway:** The `transformer` encoder had a much better loss curve, indicating that it is able to learn more about choosing the more representative sentences. However, its ROUGE scores are nearly identical to the `linear` encoder, which suggests both encoders capture enough information to summarize. The `transformer` encoder may potentially work better on more complex datasets.

**Outliers Included:**

<img src="encoder/loss_avg_seq_mean_outliers.png" width="450" /> <img src="encoder/loss_total_outliers.png" width="450" />

**No Outliers:**

<img src="encoder/loss_avg_seq_sum.png" width="450" /> <img src="encoder/loss_avg_seq_mean.png" width="450" />

<img src="encoder/loss_total_norm_batch.png" width="450" /> <img src="encoder/loss_avg.png" width="450" />

<img src="encoder/loss_total.png" width="450" /> <img src="encoder/loss_avg_seq_mean_val_only.png" width="450" />

**Relative Time:**

<img src="encoder/loss_avg_seq_mean_reltime.png" width="450" />