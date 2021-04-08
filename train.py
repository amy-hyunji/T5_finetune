import os
import logging 
import argparse
import textwrap 
import wandb
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import T5Tokenizer
from dataloader import Trivia_QA_Closedbook, Hotpot_QA_Closedbook, Complex_QA_Closedbook
from pytorch_lightning.loggers import WandbLogger
from model import T5FineTuner
from utils import set_seed, LoggingCallback

logger = logging.getLogger(__name__)

args_dict = dict(
    wandb_key = "",
    dataset = "complex",
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-large',
    tokenizer_name_or_path='t5-large',
    add_all=False,
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

set_seed(args_dict['seed'])  # 42

assert (args_dict['dataset'] in ['hotpot', 'trivia', 'complex'])

_model_name = args_dict['model_name_or_path'].split("/")
if (len(_model_name)>1):
    if args_dict['add_all']:
        sub_name = f'add_all_{_model_name[-2]}_{_model_name[-1]}'
    else:
        sub_name = f'{_model_name[-2]}_{_model_name[-1]}'

else:
    if args_dict['add_all']:
        sub_name = f"add_all_{_model_name[-1]}"
    else:
        sub_name = _model_name[-1]

if args_dict['dataset'] == "trivia":
    args_dict.update({'output_dir': f"{args_dict['seed']}_{sub_name}_trivia_qa_closedbook", 'num_train_epochs':100,
                     'train_batch_size': 48, 'eval_batch_size': 48, 'learning_rate': 1e-3})
elif args_dict["dataset"] == "hotpot":
    args_dict.update({'output_dir': f"{args_dict['seed']}_{sub_name}_hotpot_qa_closedbook", 'num_train_epochs': 100, 
                    'train_batch_size': 48, 'eval_batch_size': 48, 'learning_rate': 1e-3}) 
                    #"resume_from_checkpoint": 'checkpointcheckpoint_ckpt_epoch_19.ckpt'})
elif args_dict['dataset'] == "complex":
    args_dict.update({'output_dir': f"error_fix_{args_dict['seed']}_{sub_name}_complex_qa_closedbook", 'num_train_epochs':100,
                     'train_batch_size': 48, 'eval_batch_size': 48, 'learning_rate': 1e-3})

args = argparse.Namespace(**args_dict)
print(args_dict)

## Define Checkpoint function
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="em_score", mode="max", save_top_k=1
)

wandb_logger = WandbLogger(project='closedbook-T5')

## If resuming from checkpoint, add an arg resume_from_checkpoint
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    #early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    resume_from_checkpoint=args.resume_from_checkpoint,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    val_check_interval=args.val_check_interval,
    logger=wandb_logger,
    callbacks=[LoggingCallback(logger)],
)

model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

print("### Saving output ###")
for split in ['validation']:
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    if args_dict['dataset'] == "trivia":
        dataset = Trivia_QA_Closedbook(tokenizer, split, None, 25, 10, False)
    elif args_dict['dataset'] == "hotpot":
        dataset = Hotpot_QA_Closedbook(tokenizer, split, None, 25, 10, False)
    elif args_dict['dataset'] == "complex":
        dataset = Complex_QA_Closedbook(tokenizer, split, None, 25, 10, False)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    it = iter(loader)

    batch = next(it)
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

    dec = [tokenizer.decode(ids) for ids in outs]

    texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
    targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

    f = open(os.path.join(args.output_dir, f"output_{split}.txt"), "w")

    for i in range(len(texts)):
        lines = textwrap.wrap(f"{args.dataset} Question:\n%s\n" % texts[i], width=100)
        print("\n".join(lines))
        print("\nActual Answer: %s" % targets[i])
        print("\nPredicted Answer from T5: %s" % dec[i])
        print("=====================================================================\n")
        f.write(f"\n{lines}")
        f.write(f"\nActual Answer: {targets[i]}")
        f.write(f"\nPredict Answer: {dec[i]}")

    f.close()
