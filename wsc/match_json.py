import json
import pandas as pd

with open('../data/switch_labels.json', 'r') as f:
    j = json.load(f)

id_dict = {}
csv = pd.read_csv('../data/wsc_data/new_test.tsv', sep='\t')
for n, item in csv.iterrows():
    correct_answer = item['answer_a'] if item['correct_answer'] == 'A' else item['answer_b']
    matched = False
    for entry in j:
        #if entry['correct_answer'].lower() == correct_answer:
        if entry['index'] == n:
            print(entry['is_switchable'])
            matched = True
            break

    if not matched:
        print()


