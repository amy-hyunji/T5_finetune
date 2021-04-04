"""
create dataframe with same answers in train, val, pred
"""

import json
import os
import pandas as pd 

trainfile = open("hotpot/hotpot_train_v1.1.json")
valfile = open("hotpot/hotpot_dev_fullwiki_v1.json")

trainjson = json.load(trainfile)
valjson = json.load(valfile)

pred_df = pd.read_csv("hotpot_t5_large_split.csv")
#pred_df = {'question': _pred_df['question'][:30], 'answer': _pred_df['answer'][:30], 'predict': _pred_df['predict'][:30]}
#pred_df = pd.DataFrame(pred_df)

if not os.path.exists("./hotpot/train.csv") and not os.path.exists("./hotpot/val.csv"):
    train_df = {'t_question': [], 'answer': []}
    val_df = {'question': [], 'answer': []}

    for _train in trainjson:
        q = _train['question']
        a = _train['answer']
        train_df['t_question'].append(q)
        train_df['answer'].append(a)
    train_df = pd.DataFrame(train_df)
    #train_df.to_csv("./hotpot/train.csv")
    print("Done saving hotpot/train.csv")

    for _val in valjson:
        q = _val['question']
        a = _val['answer']
        val_df['question'].append(q)
        val_df['answer'].append(a)
    val_df = pd.DataFrame(val_df)
    #val_df.to_csv("./hotpot/val.csv")
    print("Done saving hotpot/val.csv")
else:
    train_df = pd.read_csv("./hotpot/train.csv")
    val_df = pd.read_csv("./hotpot/val.csv")

# check the intesection between the two
print(f"# of train: {len(train_df['answer'])}")
print(f"# of val: {len(val_df['answer'])}")
print(f"# of pred: {len(pred_df['answer'])}")
print(" ")
int_df = pd.merge(pred_df, val_df, how='inner', on=['answer', 'question'])
print(f"key in intersect btw val and pred: {int_df.keys()}")
print(f"intersect btw val and pred: {len(int_df['answer'])}")
int_df = pd.merge(int_df, train_df, how='inner', on='answer')
print(f"intersect btw int and train: {len(int_df['answer'])}")
#int_df.to_csv("hotpot_t5_large_split_int.csv")

print(f"total len: {len(int_df['answer'])}")
"""
em = 0
for _valq, _predq, _EM in zip(int_df['v_question'], int_df['question'], int_df['EM']):
    if (_valq != _predq):
        print(f"hmm.. they are different?")
        print(f"valq: {_valq}\npredq: {_predq}")
        print('')
    if (_EM == 1): em += 1
print(f"# of 1 in EM: {em}")
print(f"# of 0 in EM: {len(int_df['answer'])-em}")
"""