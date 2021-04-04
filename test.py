import torch
import argparse
import os 
import textwrap
import logging 
import pytorch_lightning as pl 
import pandas as pd

from transformers import AutoTokenizer, AutoModelWithLMHead
from dataloader import Trivia_QA_Closedbook, Hotpot_QA_Closedbook 
from model import T5FineTuner 
from utils import set_seed, LoggingCallback, exact_match_score
from torch.utils.data import Dataset, DataLoader

args_dict = dict(
    model_name_or_path = "t5-base_hotpot_qa_closedbook/best_tfmr",
    tokenizer_name_or_path = "t5-base_hotpot_qa_closedbook/best_tfmr",
    output_name = "hotpot_t5_base",
    output_dir = "",
    dataset = "hotpot",
    max_input_length=25,
    max_output_length=10,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=1e-5,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=4,
    eval_batch_size=4,
    num_train_epochs=2,
    gradient_accumulation_steps=10,
    n_gpu=1,
    resume_from_checkpoint=None, 
    val_check_interval = 0.5, 
    n_val=5000,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=101,
)

if args_dict['dataset'] == "trivia":
    args_dict.update({'num_train_epochs':150,
                     'train_batch_size': 48, 'eval_batch_size': 48, 'learning_rate': 1e-3,
                     'resume_from_checkpoint': 't5_trivia_qa_closedbook/checkpointepoch=53.ckpt'})
elif args_dict["dataset"] == "hotpot":
    args_dict.update({'num_train_epochs': 100, 
                    'train_batch_size': 48, 'eval_batch_size': 48, 'learning_rate': 1e-3, 
                    "resume_from_checkpoint": 'checkpointcheckpoint_ckpt_epoch_19.ckpt'})

args = argparse.Namespace(**args_dict)
print(args_dict)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
model = T5FineTuner(args)
model.eval()
#model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)

for split in ['validation']:
    print(f"Working in {split}")
    
    if args_dict['dataset'] == "trivia":
        dataset = Trivia_QA_Closedbook(tokenizer, split, None, 25, 10, False)
    elif args_dict['dataset'] == "hotpot":
        dataset = Hotpot_QA_Closedbook(tokenizer, split, None, 25, 10, False)

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    it = iter(loader)
    #batch = next(it)
    retdict = {'question': [], 'answer': [], 'predict': [], 'EM': []}
    
    for i, batch in enumerate(it):
        batch["source_ids"].shape

        model.to('cuda')
        outs = model.model.generate(
                    batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    use_cache=True,
                    decoder_attention_mask=batch['target_mask'].cuda(),
                    max_length=10,
                    num_beams=2,
                    early_stopping=True
                )

        dec = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in outs]

        texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in batch['source_ids']]
        targets = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in batch['target_ids']]


        print(f"# of text: {len(retdict['question'])}\nSaving as: {args.output_name}_{split}.txt")
        for i in range(len(texts)):
            lines = textwrap.wrap(f"{args.dataset} Question:\n%s\n" % texts[i], width=100)
            """
            print("\n".join(lines))
            print("\nActual Answer: %s" % targets[i])
            print("\nPredicted Answer from T5: %s" % dec[i])
            print("=====================================================================\n")
            """
            retdict['question'].append(texts[i])
            retdict['answer'].append(targets[i])
            retdict['predict'].append(dec[i])
            retdict['EM'].append(exact_match_score(dec[i], targets[i]))

    df = pd.DataFrame(retdict)
    df.to_csv(f"{args.output_name}_{split}.csv")
