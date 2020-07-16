import cython
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import pandas as pd

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from matplotlib.image import imread

from theano import scan
import theano.tensor as tt

from pymc3.distributions import continuous
from pymc3.distributions import distribution

NUM_DRAWS = 2000

K = 4
n = 100
mu_0 = {0: [10, 10],
        1: [20, 20],
        2: [22, 14],
        3: [11, 16]}

sigma_0 = np.array([[1, 0],
                    [0, 1]])

pi = [0.15, 0.25, 0.15, 0.45]

if __name__ == "__main__":
    x = {i: np.random.multivariate_normal(mean=mu_0[i], 
                                      cov=sigma_0, 
                                      size=[np.random.randint(low=15, high=50, size=None)]) for i in range(K)}

    x = []
    cluster = []

    for _ in range(n):
        x_cluster = np.random.choice([0, 1, 2, 3], size=[1,], p=pi)[0]
        x.append(np.random.multivariate_normal(mean=mu_0[x_cluster],
                                            cov=sigma_0,
                                            size=[1,]).tolist()[0])
        cluster.append(x_cluster)
        
    x = np.array(x)

    with pm.Model() as model:
        p = pm.Dirichlet("p", a=[2, 2, 2, 2])
        Sigma = pm.LKJCholeskyCov("cov", n=2, eta=2, sd_dist=pm.HalfCauchy.dist(2.5, shape=2))
        L = pm.expand_packed_triangular(2, Sigma)
        
        category = pm.Categorical("category", p=p)
        
        mean = pm.MvNormal("mean", mu=np.tile(15, 8).reshape(4, 2), cov=np.array([[10, 0], [0, 10]]), shape=(4, 2))
        obs  = pm.MvNormal("obs", mu=mean[category], chol=L, observed=x)
        
        trace = pm.sample(draws=NUM_DRAWS, chains=3, tune=2000)
        
