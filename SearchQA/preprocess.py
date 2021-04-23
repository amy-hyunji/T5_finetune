import json
import os
import pandas as pd

train_path = "/mnt/hyunji/T5-finetune/SearchQA/data_json/train"
dev_path = "/mnt/hyunji/T5-finetune/SearchQA/data_json/val"
test_path = "/mnt/hyunji/T5-finetune/SearchQA/data_json/test"

tf_dict = {'question': [], 'answer': []}
df_dict = {'question': [], 'answer': []}

train_list = os.listdir(train_path)
for elem in train_list:
    with open(os.path.join(train_path, elem)) as tf:
        data = json.loads(tf.read())
        tf_dict['question'].append(data['question'])
        tf_dict['answer'].append(data['answer'])

dev_list = os.listdir(dev_path)
for elem in dev_list:
    with open(os.path.join(dev_path, elem)) as df:
        data = json.loads(df.read())
        df_dict['question'].append(data['question']) 
        df_dict['answer'].append(data['answer']) 

test_list = os.listdir(test_path)
for elem in test_list:
    with open(os.path.join(test_path, elem)) as df:
        data = json.loads(df.read())
        df_dict['question'].append(data['question']) 
        df_dict['answer'].append(data['answer']) 


tf_df = pd.DataFrame(tf_dict)
tf_df.to_csv("/mnt/hyunji/T5-finetune/SearchQA/train.csv")
 
df_df = pd.DataFrame(df_dict)
df_df.to_csv("/mnt/hyunji/T5-finetune/SearchQA/dev.csv")
    
