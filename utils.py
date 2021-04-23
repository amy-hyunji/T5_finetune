import logging
import re
import string
import torch
import sys
import os
import random
import json
import pytorch_lightning as pl
import numpy as np
import pandas as pd


class LoggingCallback(pl.Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_validation_end(self, trainer, pl_module):
        self.logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    self.logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        self.logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir, "test_results.txt"
            )
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        self.logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        em_score += exact_match_score(prediction, ground_truth)
        subset_match_score += approx_match_score(prediction, ground_truth)

    em_score /= len(predictions)
    subset_match_score /= len(predictions)
    return em_score * 100, subset_match_score * 100


"""
train # 90447
val # 7405
"""


def load_hotpot(split):
    if split == "train":
        file = "hotpot_train_v1.1.json"
    elif split == "validation":
        file = "hotpot_dev_fullwiki_v1.json"
    elif split == "test":
        # file = "hotpot_test_fullwiki_v1.json"
        file = "hotpot_dev_fullwiki_v1.json"
    else:
        print("ERROR: check `type_path` in Hotpot_QA_closedbook")
        sys.exit(-1)
    f = open(os.path.join("hotpot", file))
    f_json = json.load(f)
    print(f"[HOTPOT] split = {split} / # of data: {len(f_json)}")

    ret_list = []
    for elem in f_json:
        q = elem["question"]
        a = elem["answer"]
        ret_list.append({"question": q, "answer": a})

    assert len(ret_list) == len(f_json)
    return ret_list


"""
train set
    * # = 27639
    * answer > 1: 8129
    * avglength(answer): 2.23
val set
    * # = 3531

add_all: add all dataset // else: skip ones with > 1
"""


def load_complex(split, add_all):
    basepath = "./complex_web_questions"
    if split == "train":
        file = os.path.join(basepath, "train.json")
    elif split == "validation":
        file = os.path.join(basepath, "dev.json")
    elif split == "test":
        # file = os.path.join(basepath, "test.json")
        file = os.path.join(basepath, "dev.json")
    else:
        print("ERROR: check `type_path` in Complex_QA_closedbook")
        sys.exit(-1)
    f = open(file)
    f_json = json.load(f)

    ret_list = []
    q_list = []
    for elem in f_json:
        q = elem["question"]
        ans_list = elem["answers"]
        if (len(ans_list) > 1) and ((not add_all) or (split == "validation")):
            # instead of skipping add the random one
            # continue
            ans_num = random.randint(0, len(ans_list) - 1)
            answer = ans_list[ans_num]["answer"]
            ret_list.append({"question": str(q), "answer": str(answer)})
            q_list.append(str(q))
            continue
        if len(ans_list) == 0:
            assert False
        for _ans in ans_list:
            aliases = _ans["aliases"]
            answer = _ans["answer"]
            ret_list.append({"question": str(q), "answer": str(answer)})
            q_list.append(str(q))

    # assert (len(set(q_list)) ==  len(f_json))

    print(f"***** [COMPLEX] split = {split} / # of data: {len(ret_list)}")
    print(" ")
    print("### Example ###")
    print(f"question: {ret_list[0]['question']}")
    print(f"answer: {ret_list[0]['answer']}")
    return ret_list


"""
run LAMA/preprocess.py first
"""


def load_lama(split, add_all):
    basepath = "./LAMA/"

    ret_list = []
    if split == "train":
        df = pd.read_csv(os.path.join(basepath, "train.csv"))
    elif split == "validation":
        df = pd.read_csv(os.path.join(basepath, "val.csv"))
    elif split == "test":
        df = pd.read_csv(os.path.join(basepath, "val.csv"))
    else:
        print("ERROR: check 'type_path` in LAMA_QA_closedbook")
        sys.exit(-1)

    ret_list = []
    for (ques, ans) in zip(df["question"], df["answer"]):
        ret_list.append({"question": str(ques), "answer": str(ans)})

    print(f"***** [LAMA] split = {split} / # of data: {len(ret_list)}")
    print(" ")
    print("### Example ###")
    print(f"question: {ret_list[0]['question']}")
    print(f"answer: {ret_list[0]['answer']}")
    return ret_list


"""
loader for qangaroo dataset
"""


def load_qangaroo(split, add_all):
    basepath = "./QAngaroo/wikihop"

    if split == "train":
        df = pd.read_csv(os.path.join(basepath, "train.csv"))
    elif split == "validation":
        df = pd.read_csv(os.path.join(basepath, "dev.csv"))
    elif split == "test":
        df = pd.read_csv(os.path.join(basepath, "dev.csv"))
    else:
        print("ERROR: check 'type_path' in QAngaroo_QA_closedbook")
        sys.exit(-1)

    ret_list = []
    for (ques, ans, cand) in zip(df["question"], df["answer"], df["candidates"]):
        ret_list.append({"question": str(ques), "answer": str(ans), "candidates": cand})

    print(f"***** [QAngaroo] split = {split} / # of data: {len(ret_list)}")
    print(" ")
    print("### Example ###")
    print(f"question: {ret_list[0]['question']}")
    print(f"answer: {ret_list[0]['answer']}")
    return ret_list


"""
loader for SearchQA dataset
run SearchQA/preprocess.py first
"""


def load_search(split, add_all):
    basepath = "/mnt/hyunji/T5-finetune/SearchQA"

    if split == "train":
        df = pd.read_csv(os.path.join(basepath, "train.csv"))
    elif split == "validation":
        df = pd.read_csv(os.path.join(basepath, "dev.csv"))
    elif split == "test":
        df = pd.read_csv(os.path.join(basepath, "dev.csv"))
    else:
        print("ERROR: check 'type_path' in Search_QA_closedbook")
        sys.exit(-1)

    ret_list = []
    for (ques, ans) in zip(df["question"], df["answer"]):
        ret_list.append({"question": str(ques), "answer": str(ans)})

    print(f"***** [Search] split = {split} / # of data: {len(ret_list)}")
    print(" ")
    print("### Example ###")
    print(f"question: {ret_list[0]['question']}")
    print(f"answer: {ret_list[0]['answer']}")
    return ret_list