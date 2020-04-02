# 1. Compute regular embeddings
# 2. Compute sentence embeddings
# 3. Run through linear layer

from argparse import ArgumentParser
import pytorch_lightning as pl
from torch import nn
from Pooling import Pooling
from data import SentencesProcessor, greedy_selection, combination_selection
from transformers import AutoConfig, AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics import acc_and_f1

class Classifier(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob=0.2):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        sent_scores = self.sigmoid(x)
        return sent_scores

class ExtractiveSummarizer(pl.LightningModule):
    def __init__(self, hparams):
        super(ExtractiveSummarizer, self).__init__()

        self.hparams = hparams

        if not hparams.embedding_model_config:
            embedding_model_config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        self.word_embedding_model = AutoModel.from_pretrained(hparams.model_name_or_path, config=hparams.embedding_model_config)

        self.pooling_model = Pooling(self.word_embedding_model.config.hidden_size,
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        self.encoder = Classifier(self.word_embedding_model.config.hidden_size)

        # Data
        self.processors = {
            "train": SentencesProcessor(name="train"),
            "valid": SentencesProcessor(name="valid"),
            "test": SentencesProcessor(name="test")
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name if hparams.tokenizer_name else hparams.model_name_or_path,
            do_lower_case=hparams.do_lower_case,
            cache_dir=hparams.data_path if hparams.data_path else None
        )

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = ArgumentParser(parents=[parent_parser])
            parser.add_argument('--model_name_or_path', type=str, default="bert-base-cased")
            parser.add_argument('--model_type', type=str, default="bert")
            parser.add_argument('--tokenizer_name', type=str, default="")
            parser.add_argument('--embedding_model_config', type=object, default=None)
            parser.add_argument('--max_seq_length', type=int, default=510)
            parser.add_argument('--oracle_mode', type=str, options=["greedy", "combination"], default="greedy")
            parser.add_argument('--data_path', type=str, required=True, default="./json_data")
            parser.add_argument('--num_threads', type=int, default=4)
            parser.add_argument("--weight_decay", default=0.0, type=float)
            parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
            parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
            parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
            parser.add_argument("--val_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
            parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size per GPU/CPU for testing.")
            return parser
    
    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None, sent_rep_token_ids=None):
        word_vectors = self.word_embedding_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

        sents_vec = word_vectors[torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)

        # sent_rep_tokens = word_vectors[:, 0, :]  # CLS token is first token
        # print(word_vectors.shape)
        # input("gogo")
        # sentence_vectors = self.pooling_model(word_vectors, sent_rep_tokens, attention_mask)
        # print(sentence_vectors.shape)
        # input("gogo")
        # sentence_scores = self.encoder(sentence_vectors)
        # print(sentence_scores.shape)
        # input("gogo")

        if labels is not None:
            num_labels = 2
            loss_fct = nn.BCELoss(reduction='none')
            print("shapes" + str(sentence_scores.shape) + "     " + str(labels.shape))
            print(sentence_scores)
            loss = loss_fct(sentence_scores, labels.float())
            sentence_scores = (loss,) + sentence_scores

        return sentence_scores

    def json_to_dataset(self, json_file, processor, oracle_mode):
        logger.info('Processing %s' % json_file)
        documents = json.load(open(json_file))
        all_sources = []
        all_ids = []
        for doc in documents: # for each document in the json file
            source, tgt = doc['src'], doc['tgt']
            # source and tgt are now arrays of sentences where each sentence is an array of tokens
            if (oracle_mode == 'greedy'):
                oracle_ids = greedy_selection(source, tgt, 3)
            elif (oracle_mode == 'combination'):
                oracle_ids = combination_selection(source, tgt, 3)
            
            all_sources.append(source)
            all_ids.append(oracle_ids)
        
        processor.add_examples(all_sources, oracle_ids=all_ids)

    def prepare_data(self):
        datasets = dict()

        for corpus_type, processor in self.processors:
            # try to load from file
            dataset = processor.load()
            if dataset: # if successfully loaded from file then continue to next processor
                datasets.append(dataset)
                continue # else continue to load and generate TensorDataset

            dataset_files = []
            for json_file in glob.glob(pjoin(self.hparams.data_path, '*' + corpus_type + '.*.json')):
                real_name = os.path.basename(json_file)
                dataset_files.append((json_file, args))

            pool = Pool(num_threads)
            for d in pool.imap(json_to_dataset, (dataset_files, processor, self.hparams.oracle_mode)):
                pass

            pool.close()
            pool.join()

            datasets[corpus_type] = processor.get_features(
                self.tokenizer,
                max_length=self.hparams.max_seq_length,
                pad_on_left=bool(self.hparams.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=self.tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                return_tensors=True
            )
        
        self.datasets = datasets

    def train_dataloader(self):
        train_dataset = self.datasets['train']
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=hparams.train_batch_size)
        self.train_dataloader = train_dataloader
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = self.datasets['valid']
        valid_sampler = RandomSampler(valid_dataset)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.val_batch_size)
        return valid_dataloader

    def test_dataloader(self):
        test_dataset = self.datasets['test']
        test_sampler = RandomSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)
        return test_dataloader

    def configure_optimizers(self):
        if hparams.max_steps > 0:
            t_total = hparams.max_steps
            hparams.max_steps = hparams.max_steps // (len(self.train_dataloader) // hparams.accumulate_grad_batches) + 1
        else:
            t_total = len(self.train_dataloader) // hparams.accumulate_grad_batches * hparams.max_steps
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": hparams.weight_decay,
            },
            {"params": [p for n, p in self.parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=hparams.learning_rate, eps=hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=hparams.warmup_steps, num_training_steps=t_total
        )
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids": batch[3]}
        outputs = model(**inputs)
        loss = outputs[0]

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch, batch_idx)
        labels = batch[2]
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids": batch[3]}
        outputs = model(**inputs)
        loss = outputs[0]

        result = acc_and_f1(outputs.cpu(), labels.cpu())

        acc = torch.tensor(result['acc'])
        f1 = torch.tensor(result['f1'])
        acc_f1 = torch.tensor(result['acc_and_f1'])

        output = OrderedDict(
            {"val_loss": loss, "val_acc": acc, "val_f1": f1, "val_acc_and_f1": acc_f1}
        )
        return output
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_val_acc_and_f1 = torch.stack([x['val_acc_and_f1'] for x in outputs]).mean()

        tqdm_dict = {
            "val_loss": avg_loss,
            "val_acc": avg_val_acc, 
            "val_f1": avg_val_f1,
            "avg_val_acc_and_f1": avg_val_acc_and_f1
        }
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": avg_loss,
        }
        return result