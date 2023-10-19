import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from tqdm import tqdm

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from nlp import load_metric
import sys
import scipy.special

from transformers import (
    AdamW,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    get_linear_schedule_with_warmup
)

import wandb
YOUR_API_KEY = '81b73f5340daf141a9f9f24d0e03191c8b70cb0f'
os.environ["WANDB_API_KEY"] = YOUR_API_KEY
wandb_logger = WandbLogger(project='MQA_Pega')


def set_seed(seed):
    random.randint(0, 4)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PegaFineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(PegaFineTuner, self).__init__()
        self.hparams = hparams
        self.model = PegasusForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = PegasusTokenizer.from_pretrained(hparams.tokenizer_name_or_path, use_fast=False)
        self.training_data = Resource(tokenizer=self.tokenizer, type_path="train", num_samples=None, input_length=self.hparams.max_input_length, output_length=self.hparams.max_output_length)
        
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
            
            
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "validation": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.cos = nn.CosineSimilarity(dim=0)
        
    
    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
            
            
    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for PEGASUS."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.proc_rank <= 0
    
    
    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    

    def compute_contrastive_loss(self, pos_h, neg_h):
        binomial_coefficient, loss = scipy.special.comb(len(pos_h), 2), torch.tensor(0.0).type_as(pos_h[0])
        
        for i in range(len(pos_h)):
            for j in range(len(pos_h)):
                if i == j: continue
                numerator, denominator = torch.exp(self.cos(pos_h[i], pos_h[j]) / self.hparams.tau).type_as(pos_h[0]), torch.tensor(0.0).type_as(pos_h[0])

                for k in range(len(neg_h)):
                    denominator += torch.exp(self.cos(pos_h[i], neg_h[k]) / self.hparams.tau)

                for k in range(len(pos_h)):
                    if i == k: continue
                    denominator += torch.exp(self.cos(pos_h[i], pos_h[k]) / self.hparams.tau)

                loss += torch.log(numerator / denominator)
        
        return -1 / scipy.special.comb(len(pos_h), 2) * loss

    def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True
    )


    def _step(self, batch, training_mode=False):
        batch_labels = batch["target_ids"]
        batch_labels[batch_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch_labels,
            decoder_attention_mask=batch['target_mask']
        )

        effective_batch_size = outputs.logits.size()[0]
        
        if training_mode:
            medical_loss, negation_loss, contrastive_loss = torch.tensor(0.0).type_as(outputs[0]), torch.tensor(0.0).type_as(outputs[0]), torch.tensor(0.0).type_as(outputs[0])

            for i in range(effective_batch_size):
                total_att = torch.mean(outputs.logits[i], 0) # reduce sequence_length

                idx = batch["id"][i].item()
                medical_terms_2, neg_uni = self.training_data[idx]['medical_terms_2'] + self.training_data[idx]['medical_terms_1'], self.training_data[idx]['neg_uni']
                source = batch_labels[i]

                # update negation_loss
                if len(neg_uni) > 0:
                    for term in neg_uni:
                        id_comb = neg_unigrams_ids[term]
                        for j in range(id_comb.size()[0]):
                            neg_id = id_comb[j].item()
                            presence_neg = (source == neg_id).nonzero(as_tuple=True)[0].tolist()

                            # corner case
                            if len(presence_neg) == 0: continue

                            for p in presence_neg:
                                medical_loss += total_att[source[p]]


                # update medical_loss
                if len(medical_terms_2) > 0:
                    for term in medical_terms_2:
                        id_comb = medical_term_ids[term]
                        for j in range(id_comb.size()[0]):
                            vocab_id = id_comb[j].item()
                            presence_vocab = (source == vocab_id).nonzero(as_tuple=True)[0].tolist()
                            
                            # corner case
                            if len(presence_vocab) == 0: continue

                            for p in presence_vocab:
                                medical_loss += total_att[source[p]]

                                # modeling the 2 tokens before the medical term
                                if p - 1 >= 0: medical_loss += total_att[source[p-1]]
                                if p - 2 >= 0: medical_loss += total_att[source[p-2]]                                


            
                # update contrastive loss
                pos_h, neg_h = {}, {}

                source_id_list_pos, source_id_list_neg = [], []
                source_att_mask_list_pos, source_att_mask_list_neg = [], []
                labels_list_pos, labels_list_neg = [], []
                d_att_mask_list_pos, d_att_mask_list_neg = [], []
                for e in self.training_data[idx]["pos_set"]:
                    labels = e["input_ids"]
                    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
                    source_id_list_pos.append(batch["source_ids"][i])
                    source_att_mask_list_pos.append(batch["source_mask"][i])
                    labels_list_pos.append(labels.squeeze())
                    d_att_mask_list_pos.append(e["attention_mask"].squeeze())

                for e in self.training_data[idx]["neg_set"]:
                    labels = e["input_ids"]
                    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
                    source_id_list_neg.append(batch["source_ids"][i])
                    source_att_mask_list_neg.append(batch["source_mask"][i])
                    labels_list_neg.append(labels.squeeze())
                    d_att_mask_list_neg.append(e["attention_mask"].squeeze())


                source_id_pos = torch.stack((source_id_list_pos))
                source_att_mask_pos = torch.stack((source_att_mask_list_pos))
                labels_pos = torch.stack((labels_list_pos))
                d_att_mask_pos = torch.stack((d_att_mask_list_pos))
                outs_pos = self(input_ids=source_id_pos, attention_mask=source_att_mask_pos, labels=labels_pos.type_as(batch_labels), decoder_attention_mask=d_att_mask_pos.type_as(batch_labels)).decoder_hidden_states[-1]

                source_id_neg = torch.stack((source_id_list_neg))
                source_att_mask_neg = torch.stack((source_att_mask_list_neg))
                labels_neg = torch.stack((labels_list_neg))
                d_att_mask_neg = torch.stack((d_att_mask_list_neg))
                outs_neg = self(input_ids=source_id_neg, attention_mask=source_att_mask_neg, labels=labels_neg.type_as(batch_labels), decoder_attention_mask=d_att_mask_neg.type_as(batch_labels)).decoder_hidden_states[-1]

                for z in range(len(self.training_data[idx]["pos_set"])):
                    outs = outs_pos[z]
                    pos_h[z] = torch.mean(outs, 0)

                for z in range(len(self.training_data[idx]["neg_set"])):
                    outs = outs_neg[z]
                    neg_h[z] = torch.mean(outs, 0)
                
                contrastive_loss += self.compute_contrastive_loss(pos_h, neg_h)

        loss = outputs[0]

        if training_mode:
            loss += self.hparams.lambda_CL * contrastive_loss / effective_batch_size
            loss -= self.hparams.lambda_medical * medical_loss / effective_batch_size

        return loss


    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def _generative_step(self, batch) :
        
        t0 = time.time()
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=168, 
            num_beams=2,
            repetition_penalty=1.5, 
            length_penalty=1.4, 
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
            
        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]  
    
        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
#         rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(self.lmap(len, generated_ids))
        base_metrics.update(gen_time=torch.tensor(gen_time).to(loss.device), gen_len=torch.tensor(summ_len).to(loss.device), preds=preds, target=target)
        # self.rouge_metric.add_batch(preds, target)
        
#         rouge_results = self.rouge_metric.compute() 
#         rouge_dict = self.parse_score(rouge_results)
#         base_metrics.update(rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])
        
        return base_metrics
    

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, training_mode=True)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)
    
  
    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x["val_loss"] for x in outputs]).mean()
        # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, weight_decay=0.0, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, using_native_amp=False):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()
  
    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict
    

    def train_dataloader(self):   
        n_samples = self.n_obs['train']
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", num_samples=n_samples, args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        validation_dataset = get_dataset(tokenizer=self.tokenizer, type_path="validation", num_samples=n_samples, args=self.hparams)

        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


class Resource(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):         
        file, dataset_list, count = "/export/home/njz5124/QS/Dataset_bug_fixed/" + type_path + ".txt", [], 0
        with open(file, 'r') as input:
            for jsonObj in input:
                patientDict, d = json.loads(jsonObj), {}
                
                d["id"] = count
                d["text"] = patientDict["question"]
                d["headline"] = patientDict["summary"]
                d["medical_terms_both"] = patientDict["2_medical"]
                d["medical_terms_one"] = patientDict["1_medical"]
                d["neg_uni"] = patientDict["neg_uni"]
                
                d["pos"] = []
                d["neg"] = []

                for name in glob.iglob("/export/home/njz5124/QS/P&N/Positive/" + str(count) + "/*.txt"):
                    with open(name, 'r', encoding='utf8') as f:
                        d["pos"].append(f.readlines()[0])

                for name in glob.iglob("/export/home/njz5124/QS/P&N/Negative/" + str(count) + "/*.txt"):
                    with open(name, 'r', encoding='utf8') as f:
                        d["neg"].append(f.readlines()[0])

                dataset_list.append(d)
                count += 1

        self.dataset = dataset_list
        if num_samples:
            self.dataset = self.dataset[:num_samples]
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
  
    def __len__(self):
        return len(self.dataset)
    
    # def clean_text(self, text):
    #     text = text.replace('\n','')
    #     text = text.replace('``', '')
    #     text = text.replace('"', '')
        
    #     return text
    
    
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", example_batch['text'])
#         input_ = self.clean_text(example_batch['text']) + " </s>"
#         target_ = self.clean_text(example_batch['headline']) + " </s>"
        
        input_ = example_batch['text']
        target_ = example_batch['headline']
        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        pos_set, neg_set = [], []
        for e in example_batch['pos']:
            pos_set.append(self.tokenizer.batch_encode_plus([e], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt"))

        for e in example_batch['neg']:
            neg_set.append(self.tokenizer.batch_encode_plus([e], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt"))
       
        return source, targets, pos_set, neg_set
  
    def __getitem__(self, index):
        source, targets, pos_set, neg_set = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        medical_terms_2 = self.dataset[index]["medical_terms_both"]
        medical_terms_1 = self.dataset[index]["medical_terms_one"]
        neg_uni = self.dataset[index]["neg_uni"]

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "pos_set": pos_set, "neg_set": neg_set, "medical_terms_2": medical_terms_2, "medical_terms_1": medical_terms_1, "neg_uni": neg_uni}




class OwnData(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):         
        file, dataset_list, count = "/export/home/njz5124/QS/Dataset_bug_fixed/" + type_path + ".txt", [], 0
        with open(file, 'r') as input:
            for jsonObj in input:
                patientDict, d = json.loads(jsonObj), {}
                
                d["id"] = count
                d["text"] = patientDict["question"]
                d["headline"] = patientDict["summary"]

                dataset_list.append(d)
                count += 1

        self.dataset = dataset_list
        if num_samples:
            self.dataset = self.dataset[:num_samples]
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
  
    def __len__(self):
        return len(self.dataset)
    
    # def clean_text(self, text):
    #     text = text.replace('\n','')
    #     text = text.replace('``', '')
    #     text = text.replace('"', '')
        
    #     return text
    
    
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", example_batch['text'])
#         input_ = self.clean_text(example_batch['text']) + " </s>"
#         target_ = self.clean_text(example_batch['headline']) + " </s>"
        
        input_ = example_batch['text']
        target_ = example_batch['headline']
        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        return source, targets
  
    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "id": self.dataset[index]["id"]}



set_seed(42)


medical_term_ids, tokenizer = {}, PegasusTokenizer.from_pretrained('google/pegasus-large', use_fast=False)
with open('/export/home/njz5124/QS/ALL_medical_term_file_train.txt', 'r', encoding='utf8') as f:
    custom_noun = f.readlines()
    for i in range(len(custom_noun)):
        medical_term = custom_noun[i].replace('\n', '')
        ids = tokenizer.batch_encode_plus([medical_term], truncation=True, return_tensors="pt")['input_ids'][0]
        # remove 1
        if ids[-1].item() == 1:
            ids = torch.cat([ids[0:ids.size()[0]-1], ids[ids.size()[0]:]])
        
        medical_term_ids[medical_term] = ids
print("Finished reading medical_term_file.txt !")


neg_unigrams, neg_unigrams_ids = ["no", "nope", "doesn't", "don't", "not"], {}
for e in neg_unigrams:
    ids = tokenizer.batch_encode_plus([e], truncation=True, return_tensors="pt")['input_ids'][0]
    # remove 1
    if ids[-1].item() == 1:
        ids = torch.cat([ids[0:ids.size()[0]-1], ids[ids.size()[0]:]])
    
    neg_unigrams_ids[e] = ids
print("Finished construction of neg_unigrams_ids!")


logger = logging.getLogger(__name__)
args_dict = dict(
    output_dir="PEGA-finetune", # path to save the checkpoints
    model_name_or_path='google/pegasus-large',
    tokenizer_name_or_path='google/pegasus-large',
    max_input_length=512,
    max_output_length=84,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=0.00003,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=600,
    train_batch_size=4,
    eval_batch_size=16,
    num_train_epochs=60,
    gradient_accumulation_steps=8,
    n_gpu=2,
    resume_from_checkpoint=None, 
    val_check_interval = 0.05, 
    n_val=1000,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
    tau=1.0,
    lambda_CL=1.0,
    lambda_medical=0.001
)


args_dict.update({'output_dir': 'pega_our_2', 'num_train_epochs':60,'train_batch_size': 4, 'eval_batch_size': 16})
args = argparse.Namespace(**args_dict)

## Define Checkpoint function
checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1)

## If resuming from checkpoint, add an arg resume_from_checkpoint
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    resume_from_checkpoint=args.resume_from_checkpoint,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    val_check_interval=args.val_check_interval,
    logger=wandb_logger,
    callbacks=[LoggingCallback()],
    sync_batchnorm=True,
    accelerator='dp'
)

def get_dataset(tokenizer, type_path, num_samples, args):
    return OwnData(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples, input_length=args.max_input_length, output_length=args.max_output_length)

model = PegaFineTuner(args)

trainer = pl.Trainer(**train_params)
print (" Training model")
trainer.fit(model)


print ("training finished")
