import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn import decomposition
from ICA_noise import FastICA


mydata = pd.read_csv("/Users/tangyehui/UG_Project/dataset/personality/UKDA-7656-tab/tab/bbc_individual_level_data_file.tab", sep ="\t")

Y = mydata.as_matrix()
X = Y[:, 34:78]

fa = sk.decomposition.FactorAnalysis(n_components=5)
fa.fit(X)
print(np.shape(X))

La = np.abs(fa.components_)

labs = [
    "Is talkative", "Tends to find fault with others",
    "Does a thorough job",
    "Is depressed, blue",
    "Is original, comes up with new ideas",
    "Is reserved",
    "Is helpful and unselfish with others",
    "Can be somewhat careless",
    "Is relaxed, handles stress well",
    "Is curious about many different things",
    "Is full of energy",
    "Starts quarrels with others",
    "Is a reliable worker",
    "Can be tense",
    "Is ingenious, a deep thinker",
    "Generates a lot of enthusiasm",
    "Has a forgiving nature",
    "Tends to be disorganized",
    "Worries a lot",
    "Has an active imagination",
    "Tends to be quiet",
    "Is generally trusting",
    "Tends to be lazy",
    "Is emotionally stable, not easily upset",
    "Is inventive",
    "Has an assertive personality",
    "Can be cold and aloof",
    "Perseveres until the task is finished",
    "Can be moody",
    "Values artistic, aesthetic experiences",
    "Is sometimes shy, inhibited",
    "Is considerate and kind to almost everyone",
    "Does things efficiently",
    "Remains calm in tense situations",
    "Prefers work that is routine",
    "Is outgoing, sociable",
    "Is sometimes rude to others",
    "Makes plans and follows through with them",
    "Gets nervous easily",
    "Likes to reflect, play with ideas",
    "Has few artistic interests",
    "Likes to cooperate with others",
    "Is easily distracted",
    "Is sophisticated in art, music, or literature"
    ]

print(len(labs))

# print(np.transpose([labs, fa.noise_variance_, np.sum(La**2,0)]))

# ica = sk.decomposition.FastICA(n_components=5)
# ica.fit(X)
# om = ica.components_
#
ica2 = FastICA(n_components=5)
ica2.fit(X)
om2 = ica2.components_


# plt.figure(figsize=(2.5,2.25))
# ii = np.array([1,6,11,16,21,26,31,36,41])
#
# plt.subplot(1,2,1)
# plt.pcolor(La[:,ii].T,cmap='bwr')
# li = [labs[i] for i in ii]
# plt.yticks(0.5+np.arange(0,len(li)), li, size='small')
# plt.xticks(0.5+np.arange(0,5),np.arange(1,6))
# plt.xlabel('Factor')
#
# plt.subplot(1, 2, 2)
# plt.pcolor(om2[:, ii].T, cmap='bwr')
# plt.yticks(0.5+np.arange(0, len(li)), [], size='small')
# plt.xticks(0.5+np.arange(0, 5), np.arange(1, 6))
# plt.xlabel('IC')
# plt.savefig('/Users/tangyehui/UG_Project/figures/factors_agree.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)


plt.figure(figsize=(2.5,2.25))
ii = np.array([3,8,13,18,23,28,33,38])

plt.subplot(1,2,1)
plt.pcolor(La[:,ii].T,cmap='bwr')
li = [labs[i] for i in ii]
plt.yticks(0.5+np.arange(0,len(li)), li, size='small')
plt.xticks(0.5+np.arange(0,5),np.arange(1,6))
plt.xlabel('Factor')

plt.subplot(1, 2, 2)
plt.pcolor(om2[:, ii].T, cmap='bwr')
plt.yticks(0.5+np.arange(0, len(li)), [], size='small')
plt.xticks(0.5+np.arange(0, 5), np.arange(1, 6))
plt.xlabel('IC')
plt.savefig('/Users/tangyehui/UG_Project/figures/factors_neuro1.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)


# plt.figure(figsize=(2.5,2.25))
# ii = np.array([0,5,10,15,20,25,30,35])
# plt.subplot(1,2,1)
# plt.pcolor(La[:,ii].T,cmap='bwr')
# li = [labs[i] for i in ii]
# plt.yticks(0.5+np.arange(0,len(li)), li, size='small')
# plt.xticks(0.5+np.arange(0,5),np.arange(1,6))
# plt.xlabel('Factor')
#
# plt.subplot(1, 2, 2)
# plt.pcolor(om2[:, ii].T, cmap='bwr')
# plt.yticks(0.5+np.arange(0, len(li)), [], size='small')
# plt.xticks(0.5+np.arange(0, 5), np.arange(1, 6))
# plt.xlabel('IC')
# plt.savefig('/Users/tangyehui/UG_Project/figures/factors_extra.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)


# plt.figure(figsize=(2.5,2.25))
# ii = np.array([4,9,14,19,24,29,34,39,43])
#
# s = 0
# for i in ii:
#     s += fa.noise_variance_[i]
# print(s/9)
# plt.subplot(1,2,1)
# plt.pcolor(La[:,ii].T,cmap='bwr')
# li = [labs[i] for i in ii]
# plt.yticks(0.5+np.arange(0,len(li)), li, size='small')
# plt.xticks(0.5+np.arange(0,5),np.arange(1,6))
# plt.xlabel('Factor')
#
# plt.subplot(1, 2, 2)
# plt.pcolor(om2[:, ii].T, cmap='bwr')
# plt.yticks(0.5+np.arange(0, len(li)), [], size='small')
# plt.xticks(0.5+np.arange(0, 5), np.arange(1, 6))
# plt.xlabel('IC')
# plt.savefig('/Users/tangyehui/UG_Project/figures/factors_open.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)


# plt.figure(figsize=(2.5,2.25))
# ii = np.array([2,7,12,17,22,27,32,37,42])
#
# s = 0
# for i in ii:
#     s += fa.noise_variance_[i]
# print(s/9)

# plt.subplot(1,2,1)
# plt.pcolor(La[:,ii].T,cmap='bwr')
# li = [labs[i] for i in ii]
# plt.yticks(0.5+np.arange(0,len(li)), li, size='small')
# plt.xticks(0.5+np.arange(0,5),np.arange(1,6))
# plt.xlabel('Factor')
#
# plt.subplot(1, 2, 2)
# plt.pcolor(om2[:, ii].T, cmap='bwr')
# plt.yticks(0.5+np.arange(0, len(li)), [], size='small')
# plt.xticks(0.5+np.arange(0, 5), np.arange(1, 6))
# plt.xlabel('IC')
# plt.savefig('/Users/tangyehui/UG_Project/figures/factors_conc1.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
