import json
import os
import sys
import pandas as pd

"""
different from hotpot in that answer is a list of dict with key: ['aliases', 'answer', 'answer_id']

question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
answer: [{'aliases': ['Washington D.C.', 'Washington', 'The District', 'U.S. Capital', 'District of Columbia / Washington city', 'The District of Columbia', 'District of Columbia', 'Washington DC'], 'answer': 'Washington, D.C.', 'answer_id': 'm.0rh6k'}]

"""

def load_trainfile():
    trainfile = open("../complex_web_questions/train.json")
    trainjson = json.load(trainfile)
    traindict = {'question': [], 'answers': []}
    for _train in trainjson:
        traindict['question'].append(_train['question'])
        traindict['answer'].append(_train['answer'])
    train_df = pd.DataFrame(traindict)
    return train_df

def load_valfile():
    valfile = open("../complex_web_questions/dev.json")
    valjson =  json.load(valfile)
    valdict = {'question': [], 'answer': []}
    for _val in valjson:
        valdict['question'].append(_val['question'])
        valdict['answer'].append(_val['answer'])
    val_df = pd.DataFrame(valdict)
    return val_df



if __name__ == "__main__":
    train_df = load_trainfile()
    val_df = load_valfile()

    # load the answerfile with key: ['question', 'answer', 'predict', 'EM']

    # check the case with multiple answer - val 

