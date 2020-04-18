# Experiments

## Loss Functions

The loss function implementation can be found in [model.py](model.py) with signature `compute_loss(self, outputs, labels, mask)`. The function uses `nn.BCELoss` with `reduction="none"` and then applies 5 different reduction techniques. Special reduction methods were needed to ignore padding and operate on the multi-class-per-document approach (each input is assigned more than one of the same class) that this research uses to perform extractive summarization. See the comments throughout the function for more information. The five different reduction methods were tested with the `distilbert-base-uncased` word embedding model and the `pooling_mode` set to `sent_rep_tokens`. Training time is just under 4 hours on a Tesla P100 (3h52m average).

The `--loss_key` argument specifies the reduction method to use. It can be one of the following: `loss_total`, `loss_total_norm_batch`, `loss_avg_seq_sum`, `loss_avg_seq_mean`, `loss_avg`.

Full command used to run the tests:

```
!python main.py \
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
--val_batch_size 32 --train_batch_size 32 --test_batch_size 32
```

### Results

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

Coming soon...
