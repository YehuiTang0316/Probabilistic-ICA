import matplotlib.pyplot as plt
import numpy as np
from ICA_noise import FastICA
import pandas as pd
from sklearn.model_selection import cross_val_score

mydata = pd.read_csv("./dataset/personality/UKDA-7656-tab/tab/bbc_individual_level_data_file.tab", sep="\t")

Y = mydata.as_matrix()
X = Y[:, 34:78]

nc = np.arange(2, 20, 3)


def compute_score(X):
    ica = FastICA(algorithm='deflation')
    scores = []

    for n in nc:
        ica.n_components = n
        scores.append(np.mean(cross_val_score(ica, X)))
    return scores


ica_scores = compute_score(X)
n_components_ica = nc[np.argmax(ica_scores)]

print("best n_components by ICA = %d" % n_components_ica)
print(ica_scores)


plt.figure()
plt.plot(nc, ica_scores, 'y', label='ICA Score')
plt.axvline(n_components_ica, color='b', label='ICA CV: %d' % n_components_ica, linestyle='--')
plt.axvline(5, color='r', label='Truth', linestyle='--')
plt.xlabel('nb of components')
plt.ylabel('CV scores')
plt.legend(loc='lower right')
plt.title('ICA Model Selection')
plt.savefig('./figures/ICA_optimal#ICs.pdf')
plt.show()