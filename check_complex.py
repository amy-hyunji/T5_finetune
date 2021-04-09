import json
import os
import sys
import unicodedata
import re
import pandas as pd


"""
different from hotpot in that answer is a list of dict with key: ['aliases', 'answer', 'answer_id']

question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
answer: [{'aliases': ['Washington D.C.', 'Washington', 'The District', 'U.S. Capital', 'District of Columbia / Washington city', 'The District of Columbia', 'District of Columbia', 'Washington DC'], 'answer': 'Washington, D.C.', 'answer_id': 'm.0rh6k'}]

"""

def load_trainfile(remove_multi):
    trainfile = open("../complex_web_questions/train.json")
    trainjson = json.load(trainfile)
    traindict = {'question': [], 'answer': []}
    for _train in trainjson:
        if remove_multi and len(_train['answers'])>1:
            continue
        else:
            traindict['question'].append(_train['question'])
            traindict['answer'].append(_train['answers'])
    train_df = pd.DataFrame(traindict)
    print(f"[remove_multi: {remove_multi}] # of train question: {len(traindict['question'])}")
    return train_df

def load_valfile(remove_multi):
    valfile = open("../complex_web_questions/dev.json")
    valjson =  json.load(valfile)
    valdict = {'question': [], 'answer': []}
    for _val in valjson:
        if remove_multi and len(_val['answers'])>1:
            continue 
        else:
            valdict['question'].append(_val['question'])
            valdict['answer'].append(_val['answers'])
    val_df = pd.DataFrame(valdict)
    print(f"[remove_multi: {remove_multi}] # of val question: {len(valdict['question'])}")
    return val_df


"""
similar to compare_span_to_answer() in complex_eval

spans --> predict answer list
"""
def compare_span_to_answer(spans, answers, question):
    pre_ans = []
    for _ans in answers:
        _ans = _ans.lower().strip()
        _ans = unicodedata.normalize('NFKD', _ans).encode('ascii', 'ignore').decode(encoding="UTF-8")
        # removing common endings such as 'f.c.'
        _ans = re.sub(r'\W', ' ', _ans).lower().strip()
        # removing The, a, an from beginning of  answer as proposed by SQUAD dataset answer comparison
        if _ans.startswith('the '):
            _ans = _ans[4:]
        if _ans.startswith('a '):
            _ans= _ans[2:]
        if _ans.startswith('an '):
            _ans= _ans[3:]
        pre_ans.append(_ans)

    question = question.lower().strip()
    found_answers = []

    #exact match:
    for pre_proc_answer, answer in zip(pre_ans, answers):
        
        if answer in spans:                
            found_answers.append(answer)

        if pre_proc_answer in spans:                
            found_answers.append(pre_proc_answer)

        # year should match year.
        if question.find('year') > -1:
            year_in_answer = re.search('([1-2][0-9]{3})', answer)
            if year_in_answer is not None:
                year_in_answer = year_in_answer.group(0)
            else:
                continue

            for span in spans:
                if year_in_answer in span:
                    found_answers.append(year_in_answer)

    return list(set(found_answers))

def compute_P1(matched_answers):
    P1 = 0
    if len(matched_answers) > 0:
        P1 = 100

    return P1

"""
group the answers, predicts for same question
get the score by offical eval method
"""
def group_multi(ans_df, val_df, remove_multi):

    # what is different?
    """
    only_ans = list((set(ans_df['question']) - set(val_df['question'])))
    only_val = list((set(val_df['question']) - set(ans_df['question'])))
    df = pd.DataFrame({'only_ans': sorted(only_ans), 'only_val': sorted(only_val)})
    df.to_csv("./what_is_the_difference.csv")
    """

    retdict = {'question': [], 'answers': [], 'predicts': [], 'score': []}
    n_ques = None 
    
    #  group  predict from ans_df + add golden_answer_list
    for (_ques, _pred) in zip(ans_df['question'], ans_df['predict']):

        if n_ques is None:
            # first case
            n_ques = _ques 
            pred_list = [_pred] 
        elif (n_ques == _ques):
            # group the answer, predict
            pred_list.append(_pred)
        else:
            # add to retdict 
            retdict['question'].append(n_ques)
            retdict['predicts'].append(pred_list)
            # prepare for new question
            n_ques = _ques
            pred_list = [_pred]

    # for the last case
    retdict['question'].append(n_ques)
    retdict['predicts'].append(pred_list)

    # remove the case with multi prediction
    if remove_multi:
        single_question = []
        single_predict = []
        for ques, pred in zip(retdict['question'], retdict['predicts']):
            if (len(pred)>1):
                continue 
            else:
                single_question.append(ques)
                single_predict.append(pred)
        retdict['question'] = single_question
        retdict['predicts'] = single_predict

    assert(len(retdict['question']) == len(retdict['predicts']))
    assert (len(retdict['question']) == len(set(retdict['question'])))
    assert (len(retdict['question']) == len(val_df['question']))

    # add gold_answer_list
    val_question = list(val_df['question'])
    val_answer = list(val_df['answer'])
    
    for pred_ques, val_ques, val_ans in zip(retdict['question'], val_question, val_answer):
        gold_answer_list = []
        for elem in val_ans:
            gold_answer_list.append(elem['answer'])
            gold_answer_list += elem['aliases'] 
        
        retdict['answers'].append(gold_answer_list)

    # calculate score   
    for (_ques, _ans, _pred) in zip(retdict['question'], retdict['answers'], retdict['predicts']):
        matched_answers = compare_span_to_answer(_pred, _ans, _ques)
        curr_P1 = compute_P1(matched_answers)
        retdict['score'].append(curr_P1)

    return retdict

"""
calculate EM only with the single case
"""
def cal_EM(retdict):
    total_num = 0
    correct = 0
    aliases = 0
    for ans, pred in zip(retdict['answers'], retdict['predicts']):
        if (len(pred)>1):
            continue
        else:
            total_num += 1
            # 'answer' key is predicted
            if ans[0] == pred[0]:
                correct += 1
            # prediction inside aliases
            elif pred[0] in ans:
                aliases += 1
    
    print(f"*****[EM]*****")    
    print(f"total_num: {total_num}")
    print(f"correct: {correct} // {correct/total_num*100}")
    print(f"aliases: {aliases} // {aliases/total_num*100}")
    print(f"**************")    

def check_overlap_btw_val_train(train_df, val_df):
    train_ques = set(train_df['question'])
    val_ques = set(val_df['question'])
    train_ans = list() 
    val_ans = list() 

    for ans in train_df['answer']:
        train_ans.append(ans[0]['answer']) 

    for ans in val_df['answer']:
        val_ans.append(ans[0]['answer'])

    train_ans = set(train_ans)
    val_ans = set(val_ans)

    ques_intersect = train_ques.intersection(val_ques)
    ans_intersect = train_ans.intersection(val_ans)

    print("*** check btw train and val ***")
    print(f"[Overlap in question]: {len(ques_intersect)}")
    print(f"[Overlap in answer]: {len(ans_intersect)}")
    print(" ")

if __name__ == "__main__":

    ### parameter to change ###
    ans_df = pd.read_csv("101_complex_t5_large_test.csv")
    remove_multi = True 
    ###########################

    train_df = load_trainfile(remove_multi)
    val_df = load_valfile(remove_multi)
    check_overlap_btw_val_train(train_df, val_df)

    retdict = group_multi(ans_df, val_df, remove_multi)

    # calculate avg P1
    P1 = 0
    for p in retdict['score']:
        P1 += p
    print(f"Total # of questions: {len(retdict['score'])}")
    print(f"AVG P1: {P1/len(retdict['score'])}")

    if remove_multi:
        cal_EM(retdict)