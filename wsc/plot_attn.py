import sys
import pickle
import torch
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import seaborn as sns

# plt.set_cmap('Greys_r')
# plt.set_cmap('Purples_r')
plt.set_cmap('PRGn')
plt.axis('off')


with open('bert.dump', 'rb') as f:
    answers, indices, tuples, attentions = pickle.load(f)
    for i in answers:
        for j in answers[i]:
            answers[i][j] = np.array(answers[i][j])
    del(tuples['text_context'])

def corrbo():
    for n, experiment in enumerate(tuples.keys()):
        attn_for_corr = torch.stack(attentions[experiment]['correct']['gold']).numpy()
        probs = answers[experiment]['probs']
        probs = probs[indices[experiment]['tuples']]

        print(experiment, stats.pearsonr(attn_for_corr.mean(axis=-1).mean(axis=-1), probs[:, 0]))
        # for i in range(24):
        #     for j in range(16):
        #         p = stats.pearsonr(attn_for_corr[:, i, j], probs)[1]
        #         if p < 0.1:
        #             print(i ,j)


def diffbo():
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, squeeze=True)
    for n, experiment in enumerate(['text_original', 'text_gender', 'text_number']):
        for truth in attentions[experiment]:
            for type in attentions[experiment][truth]:
                attentions[experiment][truth][type] = torch.stack(attentions[experiment][truth][type]).mean(dim=0).numpy()

        ratio = attentions[experiment]['wrong']['gold'] / attentions[experiment]['correct']['gold']

        ax[n].axis([-1, 15.5, -1, 23.5])
        ax[n].axis('off')
        ax[n].set_title(experiment[5:])

        im = ax[n].imshow(ratio, vmin=-0.5, vmax=2.5)

    cax = fig.add_axes([ax[-1].get_position().x1 + 0.04, ax[-1].get_position().y0, 0.02, ax[-1].get_position().height])
    cb = fig.colorbar(im, cax, ticks=[-0.5, 1, 2.5])
    plt.show()

def plotbo():
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
    c = Counter()
    for n, experiment in enumerate(tuples.keys()):
        for truth in attentions[experiment]:
            for type in attentions[experiment][truth]:
                attentions[experiment][truth][type] = torch.stack(attentions[experiment][truth][type]).mean(dim=0).numpy()

        x, y = n // 5, n % 5

        current_indices = indices[experiment]['tuples']
        heat = np.zeros((24, 16))
        for layer, head in tuples[experiment]['correct']['gold_top1']:
            c.update([(layer, head)])
            heat[layer, head] += 1
        for layer, head in tuples[experiment]['wrong']['gold_top1']:
            c.subtract([(layer, head)])
            heat[layer, head] -= 1

        heat[heat < 0] = 0

        ax[x, y].axis([0, 15.5, 0, 23.5])
        ax[x, y].axis('off')
        ax[x, y].set_title(experiment[5:])
        attn_diff = attentions[experiment]['correct']['gold'] - attentions[experiment]['wrong']['gold']
        attn_diff[attn_diff < 0] = 0

        im = ax[x, y].imshow(heat)
        # im = ax[x, y].imshow(attn_diff)

    print(c)
    # plt.show()
    # plt.savefig('/tmp/diff_gold_top5')

def cosbo():
    # for n, experiment in enumerate(tuples.keys()):
    for n, experiment in enumerate(['text_original']):
        correct_cos_d = np.stack(attentions[experiment]['correct']['cos_d']).transpose(1, 0)
        wrong_cos_d = np.stack(attentions[experiment]['wrong']['cos_d']).transpose(1, 0)
        correct_cos_o = np.stack(attentions[experiment]['correct']['cos_o']).transpose(1, 0)
        wrong_cos_o = np.stack(attentions[experiment]['wrong']['cos_o']).transpose(1, 0)

        print(stats.describe(diff_d[~np.isnan(diff_d)]))


if sys.argv[1] == '--corr':
    corrbo()
elif sys.argv[1] == '--plot':
    plotbo()
elif sys.argv[1] == '--diff':
    diffbo()
elif sys.argv[1] == '--cos':
    cosbo()
else:
    print("no")
