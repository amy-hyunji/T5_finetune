import json
import os

base = "hotpot"
test_spec = "hotpot_test_fullwiki_v1.json"
val_spec = "hotpot_dev_fullwiki_v1.json"

val_f = open(os.path.join(base, val_spec))
val_json = json.load(val_f)
test_f = open(os.path.join(base, test_spec))
test_json = json.load(test_f)

print(f"key of val_json: {val_json[0].keys()}")
print(f"key of test_json: {test_json[0].keys()}")
print('')
for i in range(15):
    print(f"question: {val_json[i]['question']}")
    print(f"answer: {val_json[i]['answer']}")
    print(f"type: {val_json[i]['type']}")
"""
print(f"supporting fact : {val_json[0]['supporting_facts']}")
print("")
for i, _context in enumerate(val_json[0]['context']):
    print(f"context of {i} = {_context}")
    print('')
"""