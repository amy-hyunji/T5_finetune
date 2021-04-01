import logging
import re
import string
import torch
import sys
import os
import pytorch_lightning as pl
import numpy as np

from dataset import Trivia_QA_Closedbook, Hotpot_QA_Closedbook

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataset(tokenizer, type_path, num_samples, args):
    if args.dataset == "trivia":  
        return Trivia_QA_Closedbook(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=args.max_input_length, 
                        output_length=args.max_output_length)
    elif args.dataset == "hotpot":
        return Hotpot_QA_Closedbook(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples, input_length=args.max_input_lenth, 
                        output_length=args.max_output_length)   
    else:
        sys.exit()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def approx_match_score(prediction, ground_truth):
    answer = normalize_answer(prediction) 
    gt = normalize_answer(ground_truth)
    match = 0
    gt_words = gt.split(" ")
    for word in gt_words:
        if word in answer:
            match = 1
            return match
    return match

def calculate_scores(predictions, ground_truths):
    em_score = 0
    subset_match_score = 0
    
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        prediction = predictions[i]
        em_score +=  exact_match_score(prediction, ground_truth)
        subset_match_score += approx_match_score(prediction, ground_truth)
    
    em_score /= len(predictions)
    subset_match_score /= len(predictions)
    return em_score*100, subset_match_score*100

def load_hotpot(split):
    if split == "train":
        file = "hotpot_train_v1.1.json"
    elif split == "validation":
        file = "hotpot_dev_fullwiki_v1.json"
    elif split == "test":
        file = "hotpot_test_fullwiki_v1.json"
    else:
        print("ERROR: check `type_path` in Hotpot_QA_closedbook")
        sys.exit(-1)
    f = open(os.path.join("hotpot",file))
    f_json = json.load(f)
    print(f"[HOTPOT] split = {split} / # of data: {len(f_json)}")
    
    ret_list = []
    for elem in f_json:
        q = elem ['question']
        a = elem['answer']
        ret_list.append({'question': q, 'answer': a})

    assert(len(ret_list) == len(f_json))
    return ret_list
