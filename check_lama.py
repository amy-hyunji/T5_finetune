"""
from val.csv - check from which dataset this was from
separate the score as in the paper (check the dependency of TREx)
"""

import pandas as pd
import json
import os


def load_jsonl(input_path):
    """
    Read list of objects from a JSON lines file
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    print(f"Loaded {len(data)} records from {input_path}")
    return data


def preprocess(lama_base_path):
    filelist = ["ConceptNet", "Google_RE", "Squad", "TREx"]
    elem_num = {"ConceptNet": 0, "Google_RE": 0, "Squad": 0, "TREx": 0}

    # get all dataset and remove the duplicates
    # question_dict = {<question>: {<answer>: <type>}, ...}
    question_dict = dict()
    for file in filelist:
        for _file in os.listdir(os.path.join(lama_base_path, file)):
            data = load_jsonl(os.path.join(lama_base_path, file, _file))
            for _data in data:
                if file == "TREx":
                    # iterate through evidences
                    for elem in _data["evidences"]:
                        elem_num["TREx"] += 1
                        temp_dict = dict()
                        temp_dict[elem["obj_surface"]] = "TREx"
                        question_dict[elem["masked_sentence"]] = temp_dict
                else:
                    if len(_data["masked_sentences"]) > 1:
                        continue
                    else:
                        elem_num[file] += 1
                        temp_dict = dict()
                        temp_dict[_data["obj_label"]] = file
                        question_dict[_data["masked_sentences"][0]] = temp_dict

    print("## Before Removing DUP ##")
    print(elem_num)
    print(f"# of questions: {len(question_dict.keys())}")
    print(" ")
    print("## After Removing DUP ##")
    keys = question_dict.keys()
    print(f"# of questions: {len(set(keys))}")
    print(" ")
    return question_dict


def check_val(input_file):

    # read actual files from lama folder
    question_dict = preprocess("LAMA")

    df = pd.read_csv(input_file)

    error_num = 0
    n_dict = {"question": [], "answer": [], "type": [], "predict": [], "EM": []}
    for (ques, ans, pred, em) in zip(
        df["question"], df["answer"], df["predict"], df["EM"]
    ):
        try:
            n_dict["types"].append(question_dict[ques][ans])
        except:
            n_dict["types"].append(" ")
            error_num += 1
        n_dict["question"].append(ques)
        n_dict["answer"].append(ans)
        n_dict["predict"].append(pred)
        n_dict["EM"].append(em)

    after_em = 0
    for em in n_dict["EM"]:
        after_em += int(em)

    print(f"Out of {len(df['question'])} // error: {error_num}")
    print(f"[EM]: {after_em}     {after_em/len(df['EM'])*100}")
    return n_dict


"""
check number of elements for each types
types = ['ConceptNet', 'Google_RE', 'Squad', 'TREx']
"""


def check_num_types(val_dict):
    total_num = {"ConceptNet": 0, "Google_RE": 0, "Squad": 0, "TREx": 0}
    correct_num = {"ConceptNet": 0, "Google_RE": 0, "Squad": 0, "TREx": 0}
    for type, EM in zip(val_dict["type"], val_dict["EM"]):
        total_num[type] += 1
        if EM == 1:
            correct_num[type] += 1
    print("# of types in validation set: {total_num}")
    print("# of corrects for each types: {correct_num}")
    print(" ")
    for elem in total_num.keys():
        print(f"% of correct for {elem}: {correct_num[elem]/total_num[elem]*100}")
    print(" ")
    return


if __name__ == "__main__":
    val_file = "results/lama_val.csv"
    input_file = "results/101-lama-t5-large_test.csv"

    # check val file and add new column indicating the task
    # keys in val_dict = ['question', 'answer', 'type', 'predict', 'EM']
    # val_dict -> removed the ones that cannot be found in dataset (because of tokenizer..?)
    val_dict = check_val(input_file)

    # check the # of elements for each type
    check_num_types(val_dict)