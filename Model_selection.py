import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA, FactorAnalysis
from ICA_noise import FastICA
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


mydata = pd.read_csv("./dataset/personality/UKDA-7656-tab/tab/bbc_individual_level_data_file.tab", sep="\t")

Y = mydata.as_matrix()
X = Y[:, 34:78]

n_components = np.arange(2, 20, 4)
rank = 5


def compute_scores(X):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()
    # ica = FastICA(algorithm='deflation')

    pca_scores, fa_scores, ica_scores = [], [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        # ica.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))
        # ica_scores.append(np.mean(cross_val_score(ica, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))


pca_scores, fa_scores = compute_scores(X)
n_components_pca = n_components[np.argmax(pca_scores)]
n_components_fa = n_components[np.argmax(fa_scores)]
# n_components_ica = n_components[np.argmax(ica_scores)]

pca = PCA(svd_solver='full', n_components='mle')
pca.fit(X)
n_components_pca_mle = pca.n_components_

print("best n_components by PCA CV = %d" % n_components_pca)
print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
print("best n_components by PCA MLE = %d" % n_components_pca_mle)
# print("best n_components by ICA = %d" % n_components_ica)
# print(ica_scores)

plt.figure()
plt.plot(n_components, pca_scores, 'b', label='PCA scores')
plt.plot(n_components, fa_scores, 'r', label='FA scores')
# plt.plot(n_components, ica_scores, 'y', label='ICA scores')
plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa,
                linestyle='--')
# plt.axvline(n_components_pca_mle, color='k',
#                 label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')
# plt.axvline(n_components_ica, color='p', label='ICA CV: %d' % n_components_ica, linestyle='--')

# compare with other covariance estimators
# plt.axhline(shrunk_cov_score(X), color='violet',
#                 label='Shrunk Covariance MLE', linestyle='-.')
# plt.axhline(lw_score(X), color='orange',
#                 label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

plt.xlabel('nb of components')
plt.ylabel('CV scores')
plt.legend(loc='lower right')
plt.title('Model Selection')

plt.savefig('./figures/PCA_and_FA_analysis1.pdf')
plt.show()