import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import stats



with open('description_dump.pickle', 'rb') as f:
    description, indices, answers = pickle.load(f)

for experiment in description.keys():
    for polarity in description[experiment].keys():
        # stability etc
        if isinstance(description[experiment][polarity], float):
            continue

        for word_location in description[experiment][polarity].keys():
            description[experiment][polarity][word_location] = \
                np.array(description[experiment][polarity][word_location])

    answers[experiment] = np.array(answers[experiment])

dists = []
for experiment in description.keys():
    correct_original = description['text_original']['correct']['ans']
    wrong_original = description['text_original']['wrong']['ans']
    current_indices = indices[experiment]['ans']

    correct_shift = description[experiment]['correct']['ans'] - correct_original[current_indices]
    wrong_shift = description[experiment]['wrong']['ans'] - wrong_original[current_indices]

    current_answers = answers[experiment]
    assert np.sum(current_answers) / len(current_answers) == description[experiment]['accuracy']
    # subset of answers for the original dataset that matches valid ones for current experiment
    subset_score = np.sum(answers['text_original'][current_indices]) / len(answers['text_original'][current_indices])



    print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".
          format(experiment, np.mean(correct_shift), np.mean(wrong_shift),
                 100 * subset_score, 100 * description[experiment]['accuracy'], 100 * description[experiment]['stability']))

"""
plt.ylim(top=3)
plt.bar(indices, final_probs, 0.4, color='olivedrab')
plt.bar(indices + 0.4, final_enhanced, 0.4, color='darksalmon')
plt.show()
"""