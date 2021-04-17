import json
import os
import pandas as pd

"""
file: Squad - masked_sentences -> obj_label
Loaded 305 records from Squad/test.jsonl
{'masked_sentences': ['To emphasize the 50th anniversary of the Super Bowl the [MASK] color was used.'], 'obj_label': 'gold', 'id': '56be4db0acb8001400a502f0_0', 'sub_label': 'Squad'}

file: ConceptNet- masked_sentences -> obj_label
Loaded 29774 records from ConceptNet/test.jsonl
{'sub': 'alive', 'obj': 'think', 'pred': 'HasSubevent', 'masked_sentences': ['One of the things you do when you are alive is [MASK].'], 'obj_label': 'think', 'uuid': 'd4f11631dde8a43beda613ec845ff7d1'}

file: TREx - iterate through trex['evidence']: masked_sentence -> obj_surface
Loaded 969 records from TREx/P1412.jsonl
{'uuid': '4c35f15b-d0aa-44f3-9735-f00a4a515366', 'obj_uri': 'Q652', 'obj_label': 'Italian', 'sub_uri': 'Q2397918', 'sub_label': 'Iginio Ugo Tarchetti', 'predicate_id': 'P1412', 'evidences': [{'sub_surface': 'Iginio Ugo Tarchetti', 'obj_surface': 'Italian', 'masked_sentence': 'Iginio Ugo Tarchetti ([iˈdʒinjo ˈuɡo tarˈketti]; 29 June 1839 - 25 March 1869) was an [MASK] author, poet, and journalist.'}, {'sub_surface': 'Iginio Ugo Tarchetti', 'obj_surface': 'Italian', 'masked_sentence': 'Iginio Ugo Tarchetti ([iˈdʒinjo ˈuɡo tarˈketti]; 29 June 1839 - 25 March 1869) was an [MASK] author, poet, and journalist.'}]}

file: Google_RE - masked_sentences -> obj_label
Loaded 1825 records from Google_RE/date_of_birth_test.jsonl
{'pred': '/people/person/date_of_birth', 'sub': '/m/09gb0bw', 'obj': '1941', 'evidences': [{'url': 'http://en.wikipedia.org/wiki/Peter_F._Martin', 'snippet': "Peter F. Martin (born 1941) is an American politician who is a Democratic member of the Rhode Island House of Representatives. He has represented the 75th District Newport since 6 January 2009. He is currently serves on the House Committees on Judiciary, Municipal Government, and Veteran's Affairs. During his first term of office he served on the House Committees on Small Business and Separation of Powers & Government Oversight. In August 2010, Representative Martin was appointed as a Commissioner on the Atlantic States Marine Fisheries Commission", 'considered_sentences': ['Peter F Martin (born 1941) is an American politician who is a Democratic member of the Rhode Island House of Representatives .']}], 'judgments': [{'rater': '18349444711114572460', 'judgment': 'yes'}, {'rater': '17595829233063766365', 'judgment': 'yes'}, {'rater': '4593294093459651288', 'judgment': 'yes'}, {'rater': '7387074196865291426', 'judgment': 'yes'}, {'rater': '17154471385681223613', 'judgment': 'yes'}], 'sub_w': None, 'sub_label': 'Peter F. Martin', 'sub_aliases': [], 'obj_w': None, 'obj_label': '1941', 'obj_aliases': [], 'uuid': '18af2dac-21d3-4c42-aff5-c247f245e203', 'masked_sentences': ['Peter F Martin (born [MASK]) is an American politician who is a Democratic member of the Rhode Island House of Representatives .']}
"""

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

if __name__ == "__main__":
    filelist = ['ConceptNet', 'Google_RE', 'Squad', 'TREx']
    elem_num = {'ConceptNet': 0, 'Google_RE': 0, 'Squad': 0, 'TREx': 0}
    
    # get all dataset - first get only the questions to remove duplicates!
    question_list = []
    question_answer_set = dict()
    question_filetype_set = dict()
    for file in filelist:
        for _file in os.listdir(file):
            data = load_jsonl(os.path.join(file, _file))
            for _data in data:
                if file == "TREx":
                    # iterate through 'evidences'
                    for elem in _data['evidences']:
                        elem_num['TREx'] += 1
                        question_list.append(elem['masked_sentence'])
                        question_answer_set[elem['masked_sentence']] = elem['obj_surface']
                        question_filetype_set[elem['masked_sentence']] = "TREx"
                else:
                    # cases with multiple masked_sentences exists - remove: too long!
                    if (len(_data['masked_sentences'])>1):
                        continue
                    else:
                        elem_num[file] += 1
                        question_list.append(_data['masked_sentences'][0])
                        question_answer_set[_data['masked_sentences'][0]] = _data['obj_label']
                        question_filetype_set[_data['masked_sentences'][0]] = file 
    
    print(elem_num)
    
    print(f"Before removing dup: {len(question_list)}") #1339420
    question_list = list(set(question_list))
    print(f"After removing dup: {len(question_list)}") #880541

    answer_list = []
    type_list = []
    for ques in question_list:
        answer_list.append(question_answer_set[ques])
        type_list.append(question_filetype_set[ques])

    assert(len(answer_list) == len(question_list) == len(type_list))

    df = pd.DataFrame({'question': question_list, 'answer': answer_list, 'type': type_list})
    df = df.sample(frac=1).reset_index(drop=True)
    
    question = df['question']
    answer = df['answer']
    type = df['type']
    train_num = int(len(question)*0.9)

    train_df = pd.DataFrame({'question': question[:train_num], 'answer': answer[:train_num], 'type': type[:train_num]})
    val_df = pd.DataFrame({'question': question[train_num:], 'answer': answer[train_num:], 'type': type[train_num:]})

    # print details of train / val
    

    train_df.to_csv("train.csv")
    val_df.to_csv("val.csv")
