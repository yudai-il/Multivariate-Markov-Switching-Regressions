import pandas as pd
import numpy as np
# from.tools import *
from Multivariate_Markov_Switching_Model.tools import *
from Multivariate_Markov_Switching_Model.core import *
from Multivariate_Markov_Switching_Model.tools import _2dim
import numpy as np
import os
# os.chdir("Multivariate_Markov_Switching_Model")


"""
test A
"""


data = []

with open("Multivariate_Markov_Switching_Model/test_data/MSVARUN.txt") as f:
    for _ in f.readlines():
        data.append(_.strip().split("\t"))
data = pd.DataFrame(data)

s = data.set_index([1,0]).replace(".",np.nan)[2].astype(float).rename_axis(['year','month'])

apr = data.set_index([1,0]).replace(".",np.nan)[3].astype(float).rename_axis(['year','month'])

apriori = apr[apr.index.get_loc(("1967","7")):apr.index.get_loc(("2004","3"))].values

k_lag = 2

_ = np.log(s).diff(k_lag)*100

s = _[_.index.get_loc(("1967","2"))+k_lag:_.index.get_loc(("2004","2"))+1]

s = ((s-s.mean())/s.std()).values[:,np.newaxis]
y = s

z = generate_lagged_regressors(s,3).values
x = [1]*z.shape[0]
x = _2dim(np.array(x))

model = Markov_Multivarite_Regression(y[-z.shape[0]:],x,z,2,2,"full",apriori=None)
# model = Multivariate_Markov_Switching_Model(y[-z.shape[0]:],x,None,2,2,"full",apriori=None)

res = model.fit()



"""
test B
"""


data = []

with open("Multivariate_Markov_Switching_Model/test_data/MSVARANAS.txt") as f:
    for _ in f.readlines():
        data.append(_.strip().split("\t"))
data = pd.DataFrame(data)


data = data.replace(".",np.nan).set_index([1,0]).rename_axis(['year','month']).astype(float)

k_lag = 1
_ = np.log(data).diff(k_lag)*100

s = _[_.index.get_loc(("1984","1"))+k_lag:_.index.get_loc(("2003","1"))+1]
s = s.iloc[:,:4]

s = ((s-s.mean())/s.std()).values

x = _2dim(np.array([1]*s.shape[0]))

model = Markov_Multivarite_Regression(s,x,None,3,1,"full",apriori=None)
res = model.fit()



























"""






# res = model.fit()

self = model

param = start_params(y=self.y, x=self.x, z=self.z, k_regimes=self.k_regimes, cov_type=self.cov_type,
                     cov_switch=self.cov_switch, apriori=self.apriori, mean_variance=self.mean_variance,
                     has_delta=self.has_delta)

kwargs = {}

kwargs["y"] = model.y
kwargs["x"] = model.x
kwargs["z"] = model.z
kwargs["k_regimes"] = model.k_regimes
kwargs['apriori'] = model.apriori
kwargs['mean_variance'] = model.mean_variance
kwargs['has_delta'] = model.has_delta
kwargs["cov_switch"] = model.cov_switch
kwargs['cov_type'] = model.cov_type





_ = np.log(s).diff(-2)

_ = _

_ = s[:-10]

y = data.iloc[:,2:3].astype(float)


def generate_lagged_regressors(y, p):
    if not y.ndim == 2:
        y = format_data(y)

    _z = pd.DataFrame(np.hstack([y[i:-p + i] for i in np.arange(0, p, 1)]))
    y = pd.DataFrame(y)
    _z.columns = ["%s_%s" % (_c, p - i + 1) for i in np.arange(1, p + 1, 1) for _c in y.columns]

    if p > 0:
        warnings.warn("p>0 some Financial is dropout")

    return _z


"""


# ---------------------------------------------------------------------

"""
We plot the filtered and smoothed probabilities of a recession. 
Filtered refers to an estimate of the probability at time 𝑡 
based on data up to and including time 𝑡 (but excluding time 𝑡+1,...,𝑇). 
Smoothed refers to an estimate of the probability at time 𝑡 using all the data in the sample.
"""


"""

        # covmat_title = "Error covariance matrix"
        # 
        # var = np.hsplit(var_mat, k_regimes)
        # 
        # 
        # 
        # # error_covariance_mat = parameter_results.iloc[indices[2]:indices[3]]
        # #
        # # error_covariance_stubs = ["%s.%s"%(n,m) for i,n in enumerate(y_param_name) for j,m in enumerate(y_param_name) if j>=i]
        # #
        # # covariance_table = SimpleTable(data=error_covariance_mat.values,
        # #                   headers=["coef1", "std err", "t", "p>|t|"], stubs=error_covariance_stubs, title=covmat_title,
        # #                   txt_fmt=_fmt_params)
        # # summary.tables.append(covariance_table)

"""
import statsmodels.api as sm
import pandas as pd
import numpy as np
data = pd.read_pickle("Multivariate_Markov_Switching_Model/test_data.pkl")
model = sm.tsa.VAR(data[['M2','股市']]).fit(maxlags=3)

from statsmodels.tsa.vector_ar.irf import *
self = model

model.irf()
"""

array([[[-0.22525379,  0.00593674],
        [-0.56635031,  0.04485926]],
       [[ 0.32894306, -0.00395223],
        [-0.44499289,  0.1234111 ]],
       [[ 0.30478203,  0.00154435],
        [ 0.59459188,  0.00716239]]])

"""










