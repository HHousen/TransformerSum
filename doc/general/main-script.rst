Main Script
===========

The same script is used to train, validate, and test both extractive and abstractive models. The ``--mode`` argument switches between using :class:`~extractive.ExtractiveSummarizer` and :class:`~abstractive.AbstractiveSummarizer`, with :class:`~extractive.ExtractiveSummarizer` as the default.

.. note:: The below ``--help`` output only shows the generic commands that can be used for both extractive and abstractive models. Run the command with the ``--mode`` set to see the commands specific to each summarization technique. The ``--help`` output for each is also in this documentation: :ref:`Extractive <extractive_script_help>` and :ref:`Abstractive <abstractive_script_help>`

All training arguments can be found in the `pytorch_lightning trainer documentation <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>`_.

.. _main_script_generic_options:

Output of ``python main.py --help``:

.. code-block::

    usage: main.py [-h] [--mode {extractive,abstractive}]
                    [--default_root_dir DEFAULT_ROOT_DIR]
                    [--weights_save_path WEIGHTS_SAVE_PATH] [--learning_rate LEARNING_RATE]
                    [--min_epochs MIN_EPOCHS] [--max_epochs MAX_EPOCHS]
                    [--min_steps MIN_STEPS] [--max_steps MAX_STEPS]
                    [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
                    [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--gpus GPUS]
                    [--gradient_clip_val GRADIENT_CLIP_VAL] [--overfit_pct OVERFIT_PCT]
                    [--train_percent_check TRAIN_PERCENT_CHECK]
                    [--val_percent_check VAL_PERCENT_CHECK]
                    [--test_percent_check TEST_PERCENT_CHECK] [--amp_level AMP_LEVEL]
                    [--precision PRECISION] [--seed SEED] [--profiler]
                    [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE]
                    [--num_sanity_val_steps NUM_SANITY_VAL_STEPS]
                    [--use_logger {tensorboard,wandb}] [--wandb_project WANDB_PROJECT]
                    [--gradient_checkpointing] [--do_train] [--do_test]
                    [--load_weights LOAD_WEIGHTS]
                    [--load_from_checkpoint LOAD_FROM_CHECKPOINT]
                    [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                    [--use_custom_checkpoint_callback]
                    [--custom_checkpoint_every_n CUSTOM_CHECKPOINT_EVERY_N]
                    [--no_wandb_logger_log_model]
                    [--auto_scale_batch_size AUTO_SCALE_BATCH_SIZE] [--lr_find]
                    [--adam_epsilon ADAM_EPSILON] [--optimizer_type OPTIMIZER_TYPE]
                    [--ranger-k RANGER_K] [--warmup_steps WARMUP_STEPS]
                    [--use_scheduler USE_SCHEDULER] [--end_learning_rate END_LEARNING_RATE]
                    [--weight_decay WEIGHT_DECAY] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

        optional arguments:
        -h, --help            show this help message and exit
        --mode {extractive,abstractive}
                                Extractive or abstractive summarization training. Default is
                                'extractive'.
        --default_root_dir DEFAULT_ROOT_DIR
                                Default path for logs and weights. To use this option with the
                                `wandb` logger specify the `--no_wandb_logger_log_model` option.
        --weights_save_path WEIGHTS_SAVE_PATH
                                Where to save weights if specified. Will override
                                `--default_root_dir` for checkpoints only. Use this if for
                                whatever reason you need the checkpoints stored in a different
                                place than the logs written in `--default_root_dir`. This option
                                will override the save locations when using a custom checkpoint
                                callback, such as those created when using
                                `--use_custom_checkpoint_callback or
                                `--custom_checkpoint_every_n`. If you are using the `wandb`
                                logger, then you must also set `--no_wandb_logger_log_model` when
                                using this option. Model weights are saved with the wandb logs to
                                be uploaded to wandb.ai by default. Setting this option without
                                setting `--no_wandb_logger_log_model` effectively creates two
                                save paths, which will crash the script.
        --learning_rate LEARNING_RATE
                                The initial learning rate for the optimizer.
        --min_epochs MIN_EPOCHS
                                Limits training to a minimum number of epochs
        --max_epochs MAX_EPOCHS
                                Limits training to a max number number of epochs
        --min_steps MIN_STEPS
                                Limits training to a minimum number number of steps
        --max_steps MAX_STEPS
                                Limits training to a max number number of steps
        --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                                Accumulates grads every k batches. A single step is one gradient
                                accumulation cycle, so setting this value to 2 will cause 2
                                batches to be processed for each step.
        --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                                Check val every n train epochs.
        --gpus GPUS           Number of GPUs to train on or Which GPUs to train on. (default:
                                -1 (all gpus))
        --gradient_clip_val GRADIENT_CLIP_VAL
                                Gradient clipping value
        --overfit_pct OVERFIT_PCT
                                Uses this much data of all datasets (training, validation, test).
                                Useful for quickly debugging or trying to overfit on purpose.
        --train_percent_check TRAIN_PERCENT_CHECK
                                How much of training dataset to check. Useful when debugging or
                                testing something that happens at the end of an epoch.
        --val_percent_check VAL_PERCENT_CHECK
                                How much of validation dataset to check. Useful when debugging or
                                testing something that happens at the end of an epoch.
        --test_percent_check TEST_PERCENT_CHECK
                                How much of test dataset to check.
        --amp_level AMP_LEVEL
                                The optimization level to use (O1, O2, etc…) for 16-bit GPU
                                precision (using NVIDIA apex under the hood).
        --precision PRECISION
                                Full precision (32), half precision (16). Can be used on CPU, GPU
                                or TPUs.
        --seed SEED           Seed for reproducible results. Can negatively impact performace
                                in some cases.
        --profiler            To profile individual steps during training and assist in
                                identifying bottlenecks.
        --progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                                How often to refresh progress bar (in steps). In notebooks,
                                faster refresh rates (lower number) is known to crash them
                                because of their screen refresh rates, so raise it to 50 or more.
        --num_sanity_val_steps NUM_SANITY_VAL_STEPS
                                Sanity check runs n batches of val before starting the training
                                routine. This catches any bugs in your validation without having
                                to wait for the first validation check.
        --use_logger {tensorboard,wandb}
                                Which program to use for logging. If `wandb` is chosen then model
                                weights will automatically be uploaded to wandb.ai.
        --wandb_project WANDB_PROJECT
                                The wandb project to save training runs to if `--use_logger` is
                                set to `wandb`.
        --gradient_checkpointing
                                Enable gradient checkpointing (save memory at the expense of a
                                slower backward pass) for the word embedding model. More info: ht
                                tps://github.com/huggingface/transformers/pull/4659#issue-4248418
                                71
        --do_train            Run the training procedure.
        --do_test             Run the testing procedure.
        --load_weights LOAD_WEIGHTS
                                Loads the model weights from a given checkpoint
        --load_from_checkpoint LOAD_FROM_CHECKPOINT
                                Loads the model weights and hyperparameters from a given
                                checkpoint.
        --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                                To resume training from a specific checkpoint pass in the path
                                here. Automatically restores model, epoch, step, LR schedulers,
                                apex, etc...
        --use_custom_checkpoint_callback
                                Use the custom checkpointing callback specified in `main()` by
                                `args.checkpoint_callback`. By default this custom callback saves
                                the model every epoch and never deletes the saved weights files.
                                You can change the save path by setting the `--weights_save_path`
                                option.
        --custom_checkpoint_every_n CUSTOM_CHECKPOINT_EVERY_N
                                The number of steps between additional checkpoints. By default
                                checkpoints are saved every epoch. Setting this value will save
                                them every epoch and every N steps. This does not use the same
                                callback as `--use_custom_checkpoint_callback` but instead uses a
                                different class called `StepCheckpointCallback`. You can change
                                the save path by setting the `--weights_save_path` option.
        --no_wandb_logger_log_model
                                Only applies when using the `wandb` logger. Set this argument to
                                NOT save checkpoints in wandb directory to upload to W&B servers.
        --auto_scale_batch_size AUTO_SCALE_BATCH_SIZE
                                Auto scaling of batch size may be enabled to find the largest
                                batch size that fits into memory. Larger batch size often yields
                                better estimates of gradients, but may also result in longer
                                training time. Currently, this feature supports two modes 'power'
                                scaling and 'binsearch' scaling. In 'power' scaling, starting
                                from a batch size of 1 keeps doubling the batch size until an
                                out-of-memory (OOM) error is encountered. Setting the argument to
                                'binsearch' continues to finetune the batch size by performing a
                                binary search. 'binsearch' is the recommended option.
        --lr_find             Runs a learning rate finder algorithm (see
                                https://arxiv.org/abs/1506.01186) before any training, to find
                                optimal initial learning rate.
        --adam_epsilon ADAM_EPSILON
                                Epsilon for Adam optimizer.
        --optimizer_type OPTIMIZER_TYPE
                                Which optimizer to use: `adamw` (default), `ranger`, `qhadam`,
                                `radam`, or `adabound`.
        --ranger-k RANGER_K   Ranger (LookAhead) optimizer k value (default: 6). LookAhead
                                keeps a single extra copy of the weights, then lets the
                                internalized ‘faster’ optimizer (for Ranger, that’s RAdam)
                                explore for 5 or 6 batches. The batch interval is specified via
                                the k parameter.
        --warmup_steps WARMUP_STEPS
                                Linear warmup over warmup_steps. Only active if `--use_scheduler`
                                is set to linear.
        --use_scheduler USE_SCHEDULER
                                Three options: 1. `linear`: Use a linear schedule that inceases
                                linearly over `--warmup_steps` to `--learning_rate` then
                                decreases linearly for the rest of the training process. 2.
                                `onecycle`: Use the one cycle policy with a maximum learning rate
                                of `--learning_rate`. (default: False, don't use any scheduler)
                                3. `poly`: polynomial learning rate decay from `--learning_rate`
                                to `--end_learning_rate`
        --end_learning_rate END_LEARNING_RATE
                                The ending learning rate when `--use_scheduler` is poly.
        --weight_decay WEIGHT_DECAY
        -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                                Set the logging level (default: 'Info').
