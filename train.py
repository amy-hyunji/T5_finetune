import logging 
import argparse
import textwrap 
import wandb
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import T5Tokenizer
from dataloader import Trivia_QA_Closedboo
from pytorch_lightning.loggers import WandbLogger
from model import T5FineTuner
from utils import set_seed, LoggingCallback

set_seed(42)

logger = logging.getLogger(__name__)

args_dict = dict(
    wandb_key = "",
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
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

args_dict.update({'output_dir': 't5_trivia_qa_closedbook', 'num_train_epochs':150,
                 'train_batch_size': 48, 'eval_batch_size': 48, 'learning_rate': 1e-3,
                 'resume_from_checkpoint': 't5_trivia_qa_closedbook/checkpointepoch=53.ckpt'})
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
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    resume_from_checkpoint=args.resume_from_checkpoint,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    val_check_interval=args.val_check_interval,
    logger=wandb_logger,
    callbacks=[LoggingCallback()],
)

model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
dataset = Trivia_QA_Closedbook(tokenizer, 'validation', None, 25, 10, False)

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

for i in range(10):
    lines = textwrap.wrap("Trivia Question:\n%s\n" % texts[i], width=100)
    print("\n".join(lines))
    print("\nActual Answer: %s" % targets[i])
    print("\nPredicted Answer from T5: %s" % dec[i])
    print("=====================================================================\n")