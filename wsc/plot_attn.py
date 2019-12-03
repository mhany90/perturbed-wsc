import pandas as pd
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

plt.set_cmap('Greys_r')
# plt.set_cmap('Purples')
# plt.set_cmap('PRGn')
# plt.axis('off')


type_dict = {'gold': 'Gold referent', 'both': 'Both referents', 'discrim': 'Discriminatory tokens',
             'other': 'All other tokens', 'randtok': 'Random token'}
exp_dict = {'text_original': 'ORIG', 'text_tense': 'TEN', 'text_number': 'NUM',
            'text_gender': 'GEN', 'text_scrambled': 'SCR', 'text_adverb': 'ADV',
            'text_syn': 'SYN/NA', 'text_voice': 'VC', 'text_rel_1': 'RC'}
with open('bert.dump', 'rb') as f:
    answers, indices, tuples, attentions, accuracies, head_to_mask = pickle.load(f)
    for i in answers:
        for j in answers[i]:
            answers[i][j] = np.array(answers[i][j])
    del(tuples['text_context'])


def dropbo():
    baselines = {}
    for exp in accuracies:
        curr = (answers[exp]['gold'][indices[exp]['tuples']] == answers[exp]['pred'][indices[exp]['tuples']])
        baselines[exp] = len(curr.nonzero()[0])


    print("ext\tnormal\tstart4\tend4\tstart8\tend8")
    exps = {}
    # full = {'both': [], 'gold': [], 'discrim': [], 'other': [], 'indices': [], 'all': [], 'pert': [], 'type': []}A
    results = {'score': [], 'exp': [], 'type': [], 'direction': [], 'masked': []}
    masko = {'num_masked': [], 'head': [], 'freq': [], 'direction': []}
    for i in range(17):
        results['head_{}'.format(i)] = []

    for i in accuracies.keys():
        if i in ['text_context', 'text_freqnoun', 'text_scrambled']:
            continue
        for j in range(17):
            for type in ['discrim']:
                for direction in ['l2h']:
                    # also append baselines if j == 0
                    if j == 0:
                        total = accuracies[i]['total']
                        results['score'].append(baselines[i] / total)
                        for head_no in range(17):
                            results['head_{}'.format(head_no)].append(0)
                    else:
                        key = 'alt.{}.{}.mask_{}'.format(direction, type, j-1)
                        results['score'].append(accuracies[i][key] / total)
                        total_heads = len(head_to_mask[i][key])
                        for head_no in range(17):
                            masko['num_masked'].append(j)
                            masko['head'].append(head_no)
                            masko['freq'].append(head_to_mask[i][key].count(head_no) / total_heads)
                            masko['direction'].append(direction)
                            results['head_{}'.format(head_no)].append(head_to_mask[i][key].count(head_no) / total_heads)

                    results['direction'].append(direction)
                    results['type'].append(type_dict[type])
                    results['exp'].append(exp_dict[i])
                    # add 1 because index 0 has 1 head masked
                    results['masked'].append(j)
    for i in results:
        results[i] = np.array(results[i])

    results = pd.DataFrame(results)
    masko = pd.DataFrame(masko)
    # plt.axis('on')

    sns.set_style("white")
    p = sns.catplot(x='num_masked', y='freq', hue='head', data=masko, kind='bar', ci=None, legend=False)
    q = sns.lineplot(x='masked', y='score', hue='direction', data=results, ax=p.axes.item())
    plt.show()

    exit()

def diffbo():
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, squeeze=True)
    for n, experiment in enumerate(['text_original', 'text_gender', 'text_number']):
        for report in attentions[experiment]:
            if report in []:
                continue
            attentions[experiment][report] = torch.stack(attentions[experiment][report]).mean(dim=0).numpy()

        ratio = attentions[experiment]['wrong.gold'] / attentions[experiment]['correct.gold']

        ax[n].axis([-1, 15.5, -1, 23.5])
        ax[n].axis('off')
        ax[n].set_title(experiment[5:])

        im = ax[n].imshow(ratio, vmin=-0.5, vmax=2.5)

    cax = fig.add_axes([ax[-1].get_position().x1 + 0.04, ax[-1].get_position().y0, 0.02, ax[-1].get_position().height])
    cb = fig.colorbar(im, cax, ticks=[-0.5, 1, 2.5])
    plt.show()


def map_one(mat):
    fig, ax = plt.subplots()
    ax.axis([-0.5, 16.5, -0.5, 24.5])
    ax.axis('off')

    if (mat < 0).any():
        min = -1
        plt.set_cmap('PRGn')
    else:
        min = 0
        plt.set_cmap('Purples')

    im = ax.imshow(mat.mean(axis=0), vmin=min, vmax=1)
    plt.show()


def predbo():
    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
    del(tuples['text_scrambled'])
    del(tuples['text_freqnoun'])

    for n, experiment in enumerate(tuples.keys()):
        if experiment in ['text_scrambled', 'text_freqnoun']:
            continue

        x, y = n // 4, n % 4

        for i in attentions[experiment]:
            attentions[experiment][i] = np.stack(attentions[experiment][i])
        # gold_diff = np.stack(attentions[experiment]['all.A']) - np.stack(attentions[experiment]['all.B'])
        # mean_layer = np.expand_dims(gold_diff.mean(axis=1), axis=1)
        # gold_diff = np.append(gold_diff, mean_layer, axis=1)
        # mean_head = np.expand_dims(gold_diff.mean(axis=2), axis=2)
        # gold_diff = np.append(gold_diff, mean_head, axis=2)

        acceptable = [n for (n, _) in enumerate(attentions[experiment]['discrim']) if not np.isnan(attentions[experiment]['discrim'][n]).any()]
        for i in attentions[experiment]:
            attentions[experiment][i] = attentions[experiment][i][acceptable]

        e = attentions[experiment]

        selected_indices = indices[experiment]['tuples']
        gold = answers[experiment]['gold']

        predictions = (gold_diff > 0).astype('int')
        indexed_gold = gold[selected_indices]

        correct = (predictions.transpose(1, 2, 0) == indexed_gold).sum(axis=-1)
        total = indexed_gold.shape[-1]

        heatmap = correct / total
        # print(heatmap[12][5])

        ax[x, y].axis([-0.5, 16.5, -0.5, 24.5])
        ax[x, y].axis('off')
        ax[x, y].set_title(experiment[5:])

        im = ax[x, y].imshow(heatmap, vmin=0, vmax=1)

    plt.show()

def plotbo():
    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
    del(tuples['text_scrambled'])
    del(tuples['text_freqnoun'])
    c = Counter()
    for n, experiment in enumerate(tuples.keys()):
        for report in attentions[experiment]:
            attentions[experiment][report] = torch.stack(attentions[experiment][report]).mean(dim=0).numpy()

        x, y = n // 4, n % 4

        # current_indices = indices[experiment]['tuples']
        # heat = np.zeros((24, 16))
        # for layer, head in tuples[experiment]['correct.gold.top5']:
        #     c.update([(layer, head)])
        #     heat[layer, head] += 1
        # for layer, head in tuples[experiment]['wrong.gold.top5']:
        #     c.subtract([(layer, head)])
        #     heat[layer, head] -= 1
        #
        # heat[heat < 0] = 0

        ax[x, y].axis([0, 15.5, 0, 23.5])
        ax[x, y].axis('off')
        ax[x, y].set_title(exp_dict[experiment], fontsize=18)
        attn_diff = attentions[experiment]['correct'] - attentions[experiment]['wrong']
        # attn_diff = attn_diff_gold - attn_diff_pred
        attn_diff[attn_diff < 0] = 0
        im = ax[x, y].imshow(attn_diff, vmin=0, vmax=0.1)#, vmax=1)
        # im = ax[x, y].imshow(attn_diff)

    plt.colorbar(im, ax=ax.ravel().tolist())
    # print(c)
    plt.savefig('/tmp/attn.png')
    plt.show()
    # plt.savefig('/tmp/diff_gold_top5')

def cosbo():
    # for n, experiment in enumerate(tuples.keys()):
    for n, experiment in enumerate(['text_original']):
        correct_cos_d = np.stack(attentions[experiment]['correct.cos_d']).transpose()
        wrong_cos_d = np.stack(attentions[experiment]['wrong.cos_d']).transpose()
        correct_cos_o = np.stack(attentions[experiment]['correct.cos_o']).transpose()
        wrong_cos_o = np.stack(attentions[experiment]['wrong.cos_o']).transpose()
        diff_correct = (correct_cos_d - correct_cos_o)
        diff_wrong = (wrong_cos_d - wrong_cos_o)
        diff_d = correct_cos_d - wrong_cos_d
        diff_o = correct_cos_o - wrong_cos_o

        plt.axis('on')
        # plt.plot((diff_o).mean(axis=1), ls='-', label=experiment[5:])

        s = stats.describe(diff_d / correct_cos_d, axis=1)
        plt.errorbar(np.arange(25), s.mean, yerr=np.sqrt(s.variance))

    plt.legend(loc='best')
    plt.show()

if sys.argv[1] == '--plot':
    plotbo()
elif sys.argv[1] == '--diff':
    diffbo()
elif sys.argv[1] == '--cos':
    cosbo()
elif sys.argv[1] == '--pred':
    predbo()
elif sys.argv[1] == '--drop':
    dropbo()
