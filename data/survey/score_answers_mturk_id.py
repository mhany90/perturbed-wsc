import pandas as pd
import sys

path_to_wsc = '../wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.adverb.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')
golds =  wsc_datapoints.loc[:, 'correct_answer']


def most_frequent(List):
    return max(set(List), key = List.count)


results_filename = sys.argv[1]
results = pd.read_csv(results_filename, sep=',', error_bad_lines=False, header=0)

results_standard_filename = "mturk/standard/Batch_3753703_batch_results_standard_new.csv"
results_standard = pd.read_csv(results_standard_filename, sep=',', error_bad_lines=False, header=0)

participant_results = results.loc[:,'example.label']
ids = results.loc[:,'Input.wsc_id']

participant_standard_results = results_standard.loc[:,'example.label']
ids_standard = results_standard.loc[:,'Input.wsc_id']

correct = 0
total = 0

for wsc_id, answer in zip(ids, participant_results):
    gold = golds[wsc_id].replace('.', '').replace(' ', '')
    if gold == answer:
        correct +=1
    total += 1

print(correct, "correct")
print(total, "total")
print(correct / total, "accuracy")


correct_standard = 0
total_standard = 0

for wsc_id, answer in zip(ids_standard, participant_standard_results):
    if wsc_id in ids.values:
        gold = golds[wsc_id].replace('.', '').replace(' ', '')
        if gold == answer:
            correct_standard +=1
        total_standard += 1

print(correct_standard, "correct standard")
print(total_standard, "total standard")
print(correct_standard / total, "accuracy standard")

#find stability
participant_results_ids = {id: result for (id, result) in zip(ids, participant_results)}
match, total = 0, 0
for wsc_id, answer in zip(ids_standard, participant_standard_results):
    if wsc_id in ids.values:
        participant_result = participant_results_ids[wsc_id]
        if participant_result == answer:
            match +=1
        total += 1

stability = match / total
print(stability, " : stability")
print('\n')

#maj vote
votes = []
correct = 0
total = 0
majority_votes_enhanced = {}
for wsc_id, answer in zip(ids, participant_results):
    votes.append(answer)
    if len(votes) == 3:
        majority = most_frequent(votes).replace('.', '').replace(' ', '')
        majority_votes_enhanced[wsc_id] = majority
        gold = golds[wsc_id].replace('.', '').replace(' ', '')
        if gold == majority:
            correct +=1
        total += 1
        votes = []

print(correct, "correct MV")
print(total, "total MV")
print(correct / total, "accuracy MV")

correct_standard = 0
total_standard = 0
majority_votes_standard = {}
for wsc_id, answer in zip(ids_standard, participant_standard_results):
    if wsc_id in ids.values:
        votes.append(answer)
        if len(votes) == 3:
            majority = most_frequent(votes).replace('.', '').replace(' ', '')
            majority_votes_standard[wsc_id] = majority
            gold = golds[wsc_id].replace('.', '').replace(' ', '')
            if gold == majority:
                correct_standard += 1
            total_standard += 1
            votes = []

print(correct_standard, "correct standard MV")
print(total_standard, "total standard MV")
print(correct_standard / total, "accuracy standard MV")

#find stability mv
match = 0
total = 0
for wsc_id, answer_enhanced in majority_votes_enhanced.items():
    answer_standard = majority_votes_standard[wsc_id]
    if answer_standard == answer_enhanced:
        match += 1
    total += 1

stability = match / total
print(stability, " : stability MV")
print('\n')