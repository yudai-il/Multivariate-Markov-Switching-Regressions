from Multivariate_Markov_Switching_Model.tools import *
from Multivariate_Markov_Switching_Model.tools import _2dim
import statsmodels.api as sm
from Multivariate_Markov_Switching_Model.summaries import *


class Markov_Multivarite_Regression(object):
    def __init__(self,y,x,z,k_regimes,variance_regimes,covariance_type,apriori,**kwargs):
        self.original_y = y
        self.original_x = x
        self.original_z = z

        self.mean_variance = False

        if z is None and (len(np.unique(x)) == 1 and x.flatten()[0] == 1):
            self.mean_variance = True

        self.covariance_type = covariance_type
        self.variance_regimes = variance_regimes
        self.apriori = apriori
        self.k_regimes = k_regimes
        self.has_delta = True

        self.x,self.y,self.z = Data(self.original_y,self.original_x,self.original_z,self.mean_variance)()

        if len(np.unique(self.z)) == 1 and self.z.flatten()[0] == 0:
            self.has_delta = False

        self.nobs,self.neqs_y = self.y.shape
        self.neqs_x = self.x.shape[1]
        self.neqs_z = self.z.shape[1]
        self.indices,self.cov_obs = slicing_parameter(self.k_regimes,self.neqs_y,self.neqs_z,
                                                          self.neqs_x,self.covariance_type,self.variance_regimes)
        self.start_params = None
        if not isinstance(self.apriori,(pd.Series,np.ndarray)):
            self.apriori = clustering(self.y,self.k_regimes)

        self.kwargs = kwargs
        self.results = {'status': -1}
        self.parameters = {"nobs":self.nobs,"neqs_y":self.neqs_y,"neqs_x":self.neqs_x,
                       "neqs_z":self.neqs_z,"k_regimes":self.k_regimes,
                       "mean_variance":self.mean_variance,"covariance_type":covariance_type,"variance_regimes":variance_regimes,
                           "indices":self.indices}
        self.current_params = None


    def fit(self):
        param = start_params(y=self.y,x=self.x,z=self.z,k_regimes=self.k_regimes,covariance_type=self.covariance_type,
                             variance_regimes=self.variance_regimes,apriori=self.apriori,mean_variance=self.mean_variance,has_delta=self.has_delta)

        self.start_params = param
        import scipy.optimize as opt

        init_param = param.squeeze()

        maxiters = self.kwargs.setdefault("maxiters",1e+3)
        epsilon=self.kwargs.setdefault("eps",np.sqrt(np.finfo(float).eps))
        # epsilon = self.kwargs.setdefault('epsilon', np.sqrt(np.finfo(float).eps))

        gtol = self.kwargs.setdefault("gtol",1.0000000000000001e-05)
        norm = self.kwargs.setdefault('norm', np.Inf)

        proper_input = isinstance(maxiters,float) == isinstance(epsilon,float) == isinstance(gtol,float)
        if not proper_input:
            raise Exception("pls enter the correct input")

        options = {"maxiter":maxiters,"eps":epsilon,"gtol":gtol,"norm":norm}
        # options = {"maxiter":maxiters,"gtol":gtol}

        # gtol = self.kwargs.setdefault('gtol', 1.0000000000000001e-05)
        # norm = self.kwargs.setdefault('norm', np.Inf)
        # epsilon = self.kwargs.setdefault('epsilon', 1.4901161193847656e-08)

        rounds = 0

        while True:
            rounds += 1
            # print(rounds)
            # ,callback=progress_bar(maxiters)
            res = opt.minimize(self.llf,init_param,method="BFGS",options=options)
            self.results["unconstrained"] = res.x[:,None]
            self.results['likelihoods'] = res.fun

            if res.status == 0:
                final_parameters = res.x

                if final_parameters.ndim <2:
                    final_parameters = final_parameters[:,None]
                parameter = self.transform_params(final_parameters)
                self.results['constrained'] = parameter
                self.results['jac'] = res.jac
                self.results['status'] = 0
                return parameter

            elif res.status == 2:
                init_param = res.x
                options['eps'] *= (options['gtol'])
            else:
                print(res)
                raise Exception("something wrong pls check ")

            if rounds>=10:
                print("out of space")
                return res

    def llf(self,param):

        param = _2dim(param)

        parameter = self.transform_params(param)

        results = self._filter(parameter)
        return results

    def convert_param(self,param):
        """transforms a vector of parameters in transition probability and ergodic matrices
            beta coefficient and var-cov matrices
        """

        cut_off = self.indices
        from Multivariate_Markov_Switching_Model.tools import _p_transition,_cov_mat,_p_ergodic

        p_trans = _p_transition(param[cut_off[0]:cut_off[1]], self.k_regimes)
        p_ergodic = _p_ergodic(p_trans, self.k_regimes)

        b = param[cut_off[1]:cut_off[2]].reshape(self.neqs_x, self.neqs_y * self.k_regimes)
        sig_mat, inv_sig_mat, det_inv_sig_mat \
            = _cov_mat(param[cut_off[2]:cut_off[3]].reshape(self.cov_obs, self.variance_regimes), self.neqs_y, self.covariance_type, self.variance_regimes)

        d = np.kron(np.ones((1, self.k_regimes)), param[cut_off[3]:cut_off[4]].reshape(self.neqs_z, self.neqs_y)) if self.has_delta else 0

        return p_ergodic, p_trans, b, d, sig_mat, inv_sig_mat, det_inv_sig_mat

    def _filter(self,parameter):
        """apply hamilton filter"""

        p_j, p_ij, b, d, var_mat, inv_var_mat,det_inv_var_mat = self.convert_param(parameter)
        self.current_params = parameter
        if np.sometrue(np.greater_equal(p_ij,1)):
            return np.inf

        nobs = self.nobs
        y_hat = np.zeros((nobs,self.neqs_y))

        p_predicted_joint = np.zeros((nobs,self.k_regimes))

        joint_likelihoods = np.zeros((nobs,1))
        filtered_probabilities = np.zeros((nobs+1,self.k_regimes))
        filtered_probabilities[0,...] = p_j.T

        mu = self.x.dot(b)+self.z.dot(d)

        _y = np.kron(np.ones((1,self.k_regimes)),self.y)

        residual = _y-mu

        if np.sometrue(np.less(det_inv_var_mat,0)):
            return np.inf

        # filtered joint probabilities
        for i in np.arange(nobs):

            cond_likelihoods = self._cond_densities(residual[i,:].T,inv_var_mat,det_inv_var_mat).T
            # P(S(t)=i,Y(t)|I(t-1))
            p_predicted_joint[i,...] = p_ij.dot(filtered_probabilities[i,...])

            tmp = cond_likelihoods*p_predicted_joint[i,...]
            joint_likelihoods[i] = tmp.sum()

            if np.isnan(joint_likelihoods[i]):
                return np.inf

            filtered_probabilities[i+1,...] = tmp/joint_likelihoods[i]

            y_hat[i,:] = filtered_probabilities[i+1,...].dot(mu[i,:].reshape(self.k_regimes,self.neqs_y))

        resid = self.y-y_hat
        likelihoods = -(np.log(joint_likelihoods).sum())

        if np.isnan(likelihoods):
            raise Exception("Please Check the Calculation ")

        self.results["resid"] = resid
        self.results["joint_likelihoods"] = joint_likelihoods
        self.results['filtered_probabilities'] = filtered_probabilities
        self.results['p_predicted_joint'] = p_predicted_joint

        # ,y_hat,resid,joint_likelihoods,filtered_probabilities,p_predicted_joint
        return likelihoods

    def _cond_densities(self,res,inv_var_mat, det_inv_var_mat):
        """compute the conditional densities """
        k_regimes = self.k_regimes
        neqs_y = self.neqs_y
        _resid = res.reshape(neqs_y,k_regimes).T

        if self.variance_regimes == 1:
            sig = np.kron(np.ones((1,k_regimes)),inv_var_mat)
            det_sig = np.kron(np.ones((1,k_regimes)),det_inv_var_mat).T
        else:
            sig = inv_var_mat
            det_sig = det_inv_var_mat[:,np.newaxis]

        _resid = _resid.flatten(order="F")[:,np.newaxis]
        aux = _resid*np.kron(np.eye(k_regimes),np.ones((neqs_y,1)))
        sigma = np.kron(np.eye(k_regimes),np.ones((neqs_y,neqs_y)))*(np.kron(np.ones((k_regimes,1)),sig))

        w = aux.T.dot(sigma).dot(aux)
        v = np.diag(w)[:,np.newaxis]

        eta = (1/np.sqrt(2*np.pi))**neqs_y*np.sqrt(det_sig)*np.exp(-0.5*v)

        return eta

    def transform_params(self,unconstrained):
        """create a constrained parameter g=g(theta) to enter the likelihood function"""

        unconstrained = _2dim(unconstrained)

        k_regimes = self.k_regimes
        neqs_y = self.neqs_y
        slices = self.indices
        aux_p = unconstrained[:(k_regimes-1)*k_regimes].reshape((k_regimes-1),k_regimes)
        # FIXME over flow ignore
        _ = np.seterr(over='ignore')
        B = np.exp(aux_p)

        masks = np.isinf(B)

        sumx = B.sum(axis=0)+np.ones((1,k_regimes))[0]
        aux_p = B/sumx

        aux_p = np.where(masks,1,aux_p)

        from copy import deepcopy
        c = deepcopy(unconstrained)

        # FIXME
        prob = np.hstack([aux_p.flatten()[:,np.newaxis],0.001*np.ones(((k_regimes-1)*k_regimes,1))])
        # c.extend(prob.max(axis=1))

        c[:(k_regimes-1)*k_regimes] = prob.max(axis=1)[:,np.newaxis]

        v = c[slices[2]:slices[3],:]
        aux_v = v.reshape(self.cov_obs,self.variance_regimes)
        for i in np.arange(self.variance_regimes):
            if self.covariance_type == 'full':
                _value = aux_v[:,i]

                m_aux = xpnd(_value)

                diag_mat = np.abs(np.diag(m_aux))*(np.eye(neqs_y))

                rho_mat = m_aux-diag_mat
                rho_mat = rho_constraints(rho_mat)+np.eye(neqs_y)

                m_aux = diag_mat.dot(rho_mat).dot(diag_mat)
                aux_v[:,i] = vech(m_aux)
            else:
                aux_v[:,i] = aux_v[:,i]**2
        c[slices[2]:slices[3]] = vecr(aux_v)[:,np.newaxis]
        return c

    def summary(self):
        s = MSVARResults(self)
        return s.summary()

    def smooth_probabilities(self):
        results = self.results
        p_filtered_joint = results['filtered_probabilities'][1:]
        p_predicted_joint = results['p_predicted_joint']
        parameter = results['constrained']
        p_j, p_ij, b, d, var_mat, inv_var_mat,det_inv_var_mat = self.convert_param(parameter)

        _ = convert_smooth(p_filtered_joint, p_predicted_joint, p_ij)

        return _


i=0
import time
import sys


class progress_bar(object):
    def __init__(self,iter):
        # self.max_sec = max_sec
        self.start = time.time()
        self.iter = iter

    def __call__(self, xk=None):
        global i
        i+=1

        elapsed = time.time()-self.start
        rate = i/self.iter
        percentage = rate*100

        l_bar = '{0:3.0f}%|'.format(percentage)

        bar_length,frac_bar = divmod(int(percentage), 10)
        bar = chr(0x2588) * bar_length
        frac_bar = chr(0x2590-frac_bar)

        remaining_time = elapsed/rate - elapsed

        speed = elapsed/i

        full_bar = bar + frac_bar +' ' * max(10 - bar_length, 0)
        r_bar = '| {0}/{1} [{2}<{3}, {4}s/iter]'.format(
           i, self.iter, np.round(elapsed,4),np.round(remaining_time,4),np.round(speed,4))

        bar = l_bar+full_bar+r_bar

        sys.stdout.write(bar)
        sys.stdout.write('\n')



# def filter_probabilities(parameter):
#     llf, y_hat, resid,joint_likelihoods, filtered_probabilities, p_predicted_joint = filter(parameter,K,M,covariance_type,variance_regimes,n_x,n_z)
#     ptrans_res  = convert_param(param,K,M,n_x,n_z,covariance_type,variance_regimes)
#     param, K, M, n_x, n_z, covariance_type, variance_regimes
#     PR_SMO = MSVAR_smooth(filtered_probabilities, p_predicted_joint,ptrans_res)
#
























