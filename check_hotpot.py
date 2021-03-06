import json
import os
import sys
import pandas as pd

def load_trainfile():
    trainfile = open("./hotpot/hotpot_train_v1.1.json")
    trainjson = json.load(trainfile)
    traindict = {'question': [], 'answer': []}
    for _train in trainjson:
        traindict['question'].append(_train['question']) 
        traindict['answer'].append(_train['answer']) 
    train_df = pd.DataFrame(traindict)
    return train_df   

# load val set
def load_valfile():
    valfile = open("./hotpot/hotpot_dev_fullwiki_v1.json")
    valjson = json.load(valfile)
    valdict = {'question': [], 'answer': []}
    for _val in valjson:
        valdict['question'].append(_val['question']) 
        valdict['answer'].append(_val['answer']) 
    val_df = pd.DataFrame(valdict)

    bool_num = 0 
    inside_num = 0
    for _ques, _ans in zip(val_df['question'], val_df['answer']):
        if _ans == "yes" or _ans == "no":
            bool_num += 1
        elif _ans in _ques:
            inside_num += 1
    print(f"[VAL] bool_num: {bool_num}, inside_num: {inside_num}")

    return val_df

# compare two dfs - bool case
# keys: question, answer, predict
# check overlap questions
def compare_dfs_bool(df1, df2, save=True):
    df1_question = set(list(df1['question']))
    df2_question = set(list(df2['question']))
    
    # number should not change after changing to set type
    assert (len(df1_question) == len(df1['question']))
    assert (len(df2_question) == len(df2['question']))

    print(f"Non-overlap in df1: {len(df1_question-df2_question)}/{len(df1_question)}")
    print(f"Non-overlap in df2: {len(df2_question-df1_question)}/{len(df2_question)}")

    if save:
        only_df1_list = list(df1_question-df2_question) 
        only_df2_list = list(df2_question-df1_question)

        if len(only_df1_list) > len(only_df2_list):
            for _ in range(len(only_df1_list) - len(only_df2_list)):
                only_df2_list.append('z') 
        else:
            for _ in range(len(only_df2_list) - len(only_df1_list)):
                only_df1_list.append('z') 

        only_df1_list.sort()
        only_df2_list.sort()

        temp_df = {'only_df1': only_df1_list, 'only_df2': only_df2_list}
        temp_df = pd.DataFrame(temp_df)
        temp_df.to_csv("./bool_non_overlap_question.csv")
    return 

# compare two dfs - non-bool case
# keys: question, answer, predict, include, freq
# compare by question // print avg(freq) of non-overlaps
def compare_dfs_non_bool(df1, df2, save=True):
    df1_question = list(df1['question'])
    df1_freq = list(df1['freq'])
    df1_include = list(df1['inside'])
    
    df2_question = list(df2['question'])
    df2_freq = list(df2['freq'])
    df2_include = list(df2['inside'])

    df1_only = list(set(df1_question)-set(df2_question))
    df2_only = list(set(df2_question)-set(df1_question))
    print(f"Non-overlap in df1: {len(df1_only)}/{len(df1_question)}")
    print(f"Non-overlap in df2: {len(df2_only)}/{len(df2_question)}")    
    
    # number should not change after changing to set type
    assert (len(set(df1_question)) == len(df1_question))
    assert (len(set(df2_question)) == len(df2_question))

    # calculate avg(freq) - df1 only
    df1_freq_num = 0
    df1_include_num = 0 
    for ques in df1_only:
        idx = df1_only.index(ques)
        df1_freq_num += df1_freq[idx]
        if (df1_include[idx] == "T"): df1_include_num += 1

    df2_freq_num = 0
    df2_include_num = 0 
    for ques in df2_only:
        idx = df2_only.index(ques)
        df2_freq_num += df2_freq[idx]
        if (df2_include[idx] == "T"): df2_include_num += 1

    print(f"AVG of df1_only({len(df1_only)}): {df1_freq_num/len(df1_only)}")
    print(f"AVG of df2_only({len(df2_only)}): {df2_freq_num/len(df2_only)}")
    print(f"# inside / df1_only: {df1_include_num}/{len(df1_only)}: {df1_include_num/len(df1_only)*100}")
    print(f"# inside / df2_only: {df2_include_num}/{len(df2_only)}: {df2_include_num/len(df2_only)*100}")
    
    if save:

        only_df1_list = df1_only.copy() 
        only_df2_list = df2_only.copy() 

        if len(only_df1_list) > len(only_df2_list):
            for _ in range(len(only_df1_list) - len(only_df2_list)):
                only_df2_list.append('z') 
        else:
            for _ in range(len(only_df2_list) - len(only_df1_list)):
                only_df1_list.append('z') 

        only_df1_list.sort()
        only_df2_list.sort()

        temp_df = {'only_df1': only_df1_list, 'only_df2': only_df2_list}
        temp_df = pd.DataFrame(temp_df)
        temp_df.to_csv("./not_bool_non_overlap_question.csv")
    
    return

def get_train_freq(train_df):
    # create train_df with ['answer', 'freq']
    df = train_df.groupby(['answer']).count()
    df = df.reset_index().rename(columns={"index": "answer"})
    train_df = df.rename(columns = {'question': 'freq'})
    train_exist_ans = list(train_df['answer'])
    train_exist_freq = list(train_df['freq'])
    return train_exist_ans, train_exist_freq

# check freq
def get_freq(train_exist_ans, train_exist_freq, non_bool_df):

    # check inside for each {-1, 1, +1}
    total_non_bool = {'-1': 0, '1': 0, '+1': 0}     # total case (easy+hard) from non-bool
    inside_non_bool = {'-1': 0, '1': 0, '+1': 0}    # count easy case from non-bool 

    # get freq from non_bool_df ['question', 'answer', 'predict', 'inside']
    freq_list = []
    total_freq = 0
    total_freq_num = 0
    for _ques, _ans, _pred, _inside in zip(non_bool_df['question'], non_bool_df['answer'], non_bool_df['predict'], non_bool_df['inside']):
        if (_ans in train_exist_ans):
            idx = train_exist_ans.index(_ans)
            freq_list.append(train_exist_freq[idx])

            total_freq += train_exist_freq[idx]
            total_freq_num += 1
            
            if (train_exist_freq[idx] == 1):
                total_non_bool['1'] += 1 
                if _inside == "T": inside_non_bool['1'] += 1
            else:
                total_non_bool['+1'] += 1
                if _inside == "T": inside_non_bool['+1'] += 1
        else:
            freq_list.append(-1)
            total_non_bool['-1'] += 1
            if _inside == "T": inside_non_bool['-1'] += 1
    non_bool_df['freq'] = freq_list

    print("")
    print(f"# of total_non_bool with freq as key: {total_non_bool}")
    print(f"# of inside_non_bool with freq as key: {inside_non_bool}")
    print(f"AVG freq except -1: {total_freq/total_freq_num}")
    assert ('freq' in list(non_bool_df.keys()))

    return non_bool_df 

# get em == 1 - check bool/not & inside when not bool (T/F)
def get_em_one(df):
    bool_dict = {'question': [], 'answer': [], 'predict': []}
    non_bool_dict = {'question': [], 'answer': [], 'predict': [], 'inside': []} 
    inside_num = 0
    for _ques, _ans, _pred, _EM in zip(df['question'], df['answer'], df['predict'], df['EM']):
        if _EM == 1:
            if (_ans == "yes" or _ans == "no"):
                # bool case
                bool_dict['question'].append(_ques)
                bool_dict['answer'].append(_ans)
                bool_dict['predict'].append(_pred)
            else:
                non_bool_dict['question'].append(_ques)
                non_bool_dict['answer'].append(_ans)
                non_bool_dict['predict'].append(_pred)
                if (_ans in _ques):
                    non_bool_dict['inside'].append("T")
                    inside_num += 1
                else:
                    non_bool_dict['inside'].append("F")

    bool_df = pd.DataFrame(bool_dict)
    non_bool_df = pd.DataFrame(non_bool_dict)
    print(f"# of total questions: {len(df['question'])}")
    print(f"# of correct ones: {len(bool_dict['question'])+len(non_bool_dict['question'])} // {(len(bool_dict['question'])+len(non_bool_dict['question']))/len(df['question'])*100}")
    print(f"# of bool case: {len(bool_dict['question'])}/{len(bool_dict['question'])+len(non_bool_dict['question'])}")
    print(f"# of inside case: {inside_num}/{len(non_bool_dict['question'])}")

    return bool_df, non_bool_df

if __name__ == "__main__":
    print("Load train file ..")
    train_df = load_trainfile() # for counting freq

    print("Load validation file ..")
    val_df = load_valfile()

    train_exist_ans, train_exist_freq = get_train_freq(train_df)

    print("="*40)
    # keys: [question, answer, predict, EM]
    #t5_base = pd.read_csv("hotpot_t5_base_split.csv")
    t5_base = pd.read_csv("42_t5-base_hotpot_qa_test.csv")
    print("[BASE] Get only the correct ones")
    base_bool_df, base_non_bool_df = get_em_one(t5_base)
    # get freq of answer in trainset (w/bool) : [-1, 1, +1] 
    # and count inside for each case
    base_non_bool_df = get_freq(train_exist_ans, train_exist_freq, base_non_bool_df)
    print("")

    print("="*40)
    t5_large = pd.read_csv("101_t5-base_hotpot_qa_test.csv")
    #t5_large = pd.read_csv("101_hotpot_t5_large_validation.csv")
    print("[LARGE] Get only the correct ones")
    large_bool_df, large_non_bool_df = get_em_one(t5_large)
    large_non_bool_df = get_freq(train_exist_ans, train_exist_freq, large_non_bool_df)
    print("")

    """
    print("="*40)
    t5_large_ssm = pd.read_csv("hotpot_t5_large_ssm_split.csv")
    print("[LARGE-SSM] Get only the correct ones")
    large_ssm_bool_df, large_ssm_non_bool_df = get_em_one(t5_large_ssm)
    large_ssm_non_bool_df = get_freq(train_exist_ans, train_exist_freq, large_ssm_non_bool_df)
    print("")
    """

    print("="*40)
    print("[compare BOOL case]")
    print("** BASE - LARGE **")
    compare_dfs_bool(base_bool_df, large_bool_df)
    #print("** LARGE - LARGE_SSM **")
    #compare_dfs_bool(large_bool_df, large_ssm_bool_df)

    print("="*40)
    print("[compare NON-BOOL case]")
    print("** BASE - LARGE **")
    compare_dfs_non_bool(base_non_bool_df, large_non_bool_df)
    #print("** LARGE - LARGE_SSM **")
    #compare_dfs_non_bool(large_non_bool_df, large_ssm_non_bool_df)