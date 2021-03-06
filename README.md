# Probabilistic-ICA
Python implementation and experiments of my undergraduate project — Probabilistic ICA. This implementation is based on source code of FastICA in Sklearn.

# Dataset 
The data used in the paper is BBC Big Personality Test, 2009 - 2011. You can download it from https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=7656.
To reproduce the experiment, you need to create a folder named dataset under root path and unzip the data into it.

# Usage of the algorithm
To use the PICA algorithm, please use ```from ICA_noise import FastICA``` and set sigma in function _ica_def to the proper value.

# Experiment Results 
For separation of independent source using simulated t-distributed data with random normal distributed noise of mean 0 and standard deivation 0.5,
![img](https://github.com/YehuiTang0316/Probabilistic-ICA/blob/master/display/PCA_ICA-1.png)

For big five personalities obatined by ICA and Factor Analysis,
![img](https://github.com/YehuiTang0316/Probabilistic-ICA/blob/master/display/factors_agree-1.png)
![img](https://github.com/YehuiTang0316/Probabilistic-ICA/blob/master/display/factors_conc-1.png)
![img](https://github.com/YehuiTang0316/Probabilistic-ICA/blob/master/display/factors_extra-1.png)
![img](https://github.com/YehuiTang0316/Probabilistic-ICA/blob/master/display/factors_open-1.png)
![img](https://github.com/YehuiTang0316/Probabilistic-ICA/blob/master/display/factors_neuro-1.png)

For selection of optimal number of independent componenets using PPCA, FA and PICA,
![img](https://github.com/YehuiTang0316/Probabilistic-ICA/blob/master/display/PCA_and_FA_analysis-1.png)
![img](https://github.com/YehuiTang0316/Probabilistic-ICA/blob/master/display/ICA_optimal_ICs-1.png)
