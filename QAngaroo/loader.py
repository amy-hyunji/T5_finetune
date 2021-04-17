import json
import sys
import pandas as pd

filename = "dev"
print(f"### opening.. {filename}.json")
f = open(f"./wikihop/{filename}.json")

ret_dict = {'question': [], 'answer': [], 'candidates': []}

try:
    f_json = json.load(f)
    for elem in f_json:
        ret_dict['question'].append(elem['query'])
        ret_dict['answer'].append(elem['answer'])
        ret_dict['candidates'].append(elem['candidates'])
    df = pd.DataFrame(ret_dict)
    df.to_csv(f"./wikihop/{filename}.csv")
except:
    f.close()
    print("Error!")
    f = open("./wikihop/train.json")
    lines = f.readlines()
    print(f"total # of lines: {len(lines)}")
    inside = False
    cand = False
    for i, line in enumerate(lines):
        if ("{" in line):
            #print("inside")
            inside = True
        elif inside and "query" in line and ":" in line:
            _ques = line.split('"')[-2]
            ret_dict['question'].append(_ques)
            #print(f"question: {_ques}")
        elif inside and "answer" in line and ":" in line:
            _ans = line.split('"')[-2]
            if (len(_ans)>80): continue
            if (_ans == "."): continue
            ret_dict['answer'].append(_ans)
            #print(f"answer: {_ans}")
        elif ("}" in line):
            inside = False
            if (len(ret_dict['answer']) != len(ret_dict['question'])):
                print("ERRRORRRR")
                sys.exit()
        """
        elif inside and "candidates" in line:
            cand = True
        elif inside and cand and "[" not in line and "]" not in line:
            ret_dict['candidates'].append(line.split('"')[-2])
        elif inside and cand and "]" in line:
            cand=False
        """

    print(f"# of question: {len(ret_dict['question'])}")
    print(f"# of answer: {len(ret_dict['answer'])}")
    df = pd.DataFrame(ret_dict)
    df.to_csv(f"./wikihop/{filename}.csv")