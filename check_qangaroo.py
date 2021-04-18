"""
*** What to check
1. overlap between train/val
2. EM?
3. Overlap ~ EM의 관계성
"""

import pandas as pd
import os

"""
return dict of 
1. {'answer': count} for overlap over val set itself
2. {'answer': count} for overlap over val set & train set
"""


def overlap_btw_train_val(train_file, val_file, input_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    train_question = set(list(train_df["question"]))
    val_question = set(list(val_df["question"]))

    train_answer = set(list(train_df["answer"]))
    val_answer = set(list(val_df["answer"]))

    print(
        f"[Remove dup-train-question]   Before: {len(train_df['question'])}  After: {len(list(train_question))}"
    )
    print(
        f"[Remove dup-val-question]   Before: {len(val_df['question'])}  After: {len(list(val_question))}"
    )
    print(" ")
    print(
        f"[Remove dup-train-answer]   Before: {len(train_df['question'])}  After: {len(list(train_answer))}"
    )
    print(
        f"[Remove dup-val-answer]   Before: {len(val_df['question'])}  After: {len(list(val_answer))}"
    )
    print(" ")
    question_intersect = train_question.intersection(val_question)
    answer_intersect = train_answer.intersection(val_answer)
    print(f"### Overlap in question: {len(question_intersect)}")
    print(f"### Overlap in answer: {len(answer_intersect)}")
    print(" ")

    overlap_btw_val_dict = dict()
    input_df = pd.read_csv(input_file)
    for answer in list(input_df["answer"]):
        if answer in overlap_btw_val_dict.keys():
            overlap_btw_val_dict[answer] += 1
        else:
            overlap_btw_val_dict[answer] = 0
    return overlap_btw_val_dict, list(question_intersect), list(answer_intersect)


def get_EM(input_file):
    df = pd.read_csv(input_file)
    total_num = len(df["EM"])
    score = 0
    for em in df["EM"]:
        score += int(em)

    print(f"[EM]  #: {score}    %: {score/total_num*100}")
    return score


"""
overlap_btw_val_dict --> validation set에서 answer 당 몇 번 등장했는지 알려줌

1. 1번만 등장한 애들 개수 대비 몇 개나 맞췄는가? (correct_only_one/only_one*100)
2. count of 몇 번 등장 했는지 / em==1인 answer의 개수 --> 숫자가 높으면 val set에 많이 있는 애들을 잘 맞춘다,, (count_of/total_score*100)
3. 가장 count가 많은 애 찾고, 이와 관련된 답을 가진 question은 다 맞췄는지, 몇 개나 맞췄는지 확인 
"""


def get_additional_EM(input_file, overlap_btw_val_dict, total_score):
    df = pd.read_csv(input_file)

    correct_only_one = 0
    only_one = 0
    count_of = 0
    highest_count = 0
    highest_answer = {}
    total_highest_count = 0  # just in case the all the questions in with highest count of answer all failed

    for (ans, em) in zip(df["answer"], df["EM"]):
        if overlap_btw_val_dict[ans] == 1:
            only_one += 1
            if em == 1:
                correct_only_one += 1
        if em == 1:
            count_of += overlap_btw_val_dict[ans]
            if overlap_btw_val_dict[ans] > highest_count:
                highest_count = overlap_btw_val_dict[ans]
                highest_answer = {ans: 1}
            elif overlap_btw_val_dict[ans] == highest_count:
                if ans in highest_answer.keys():
                    highest_answer[ans] += 1
                else:
                    highest_answer[ans] = 1
        if overlap_btw_val_dict[ans] > total_highest_count:
            total_highest_count = overlap_btw_val_dict[ans]

    print(
        f"[Task1] correct_only_one: {correct_only_one}   only_one: {only_one}    %: {round(correct_only_one/only_one*100, 3)}"
    )
    print(
        f"[Task2] count_of: {count_of}   total_score: {total_score}      %: {round(count_of/total_score*100, 3)}"
    )
    print(f"[Task3] highest count from total: {total_highest_count}")
    print(
        f"[Task3] highest count from the correct ones: {highest_count} and # of corrects: \n{highest_answer}"
    )


def get_intersect_EM(input_file, intersect_list, type):
    df = pd.read_csv(input_file)

    correct_em = 0
    for (ques, ans, em) in zip(df["question"], df["answer"], df["EM"]):
        if em == 1:
            if type == "question" and ques in intersect_list:
                correct_em += 1
            if type == "answer" and ans in intersect_list:
                correct_em += 1

    print(f"### [{type}]    intersect_list #: {len(intersect_list)}")
    print(
        f"### [{type}]     correct EM: {correct_em}        %: {round(correct_em/len(intersect_list)*100, 3)}"
    )
    print(" ")


if __name__ == "__main__":

    #### Change ####
    train_file = "/home/hyunji1/T5_finetune/QAngaroo/wikihop/train.csv"
    val_file = "/home/hyunji1/T5_finetune/QAngaroo/wikihop/dev.csv"
    input_file = "./results/101-qangaroo-t5-base_test.csv"
    ################

    val_ans = pd.read_csv(val_file)["answer"]
    input_ans = pd.read_csv(input_file)["answer"]

    # 1. overlap btw train/val
    # overlap_btw_val_dict = dict containing number of occurrence in val answe
    # question_intersect = list containing intersection between train/val question
    # answer_intersect = list containing intersection between train/val answer
    overlap_btw_val_dict, question_intersect, answer_intersect = overlap_btw_train_val(
        train_file, val_file, input_file
    )

    # 2. EM
    print("=== get_EM ===")
    score = get_EM(input_file)
    print(" ")
    # 3. additional EM
    print("=== get_additional_EM ===")
    get_additional_EM(input_file, overlap_btw_val_dict, score)
    print(" ")
    print("=== get_intersect_EM===")
    get_intersect_EM(input_file, question_intersect, "question")
    get_intersect_EM(input_file, answer_intersect, "answer")
