import pandas as pd
from statsmodels.tools.numdiff import approx_hess,approx_fprime
from statsmodels.iolib.summary import *
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (gen_fmt, fmt_2,
                                                fmt_params, fmt_base, fmt_2cols)


class MSVARResults:
    def __init__(self,model):
        self.model = model

    def summary(self):

        results = self.model.results
        parameters= self.model.parameters

        unconstrained = results['unconstrained']
        constrained = results['constrained']

        k_regimes = parameters.get("k_regimes")
        indices = parameters.get("indices")

        print("calculating gradian and hessian matrix..pls be patient")

        gradian = approx_fprime(unconstrained.squeeze(),self.model.transform_params)

        hess = approx_hess(unconstrained.squeeze(),self.model.llf)

        if np.linalg.det(hess)==0:
            inv_hess = np.linalg.pinv(hess)
        else:
            inv_hess = np.linalg.inv(hess)

        cov_beta = gradian.dot(inv_hess).dot(gradian.T)
        std_err = np.sqrt(np.diag(cov_beta))
        # cor_beta = cov_beta/std_err/std_err.T
        t_student = constrained.squeeze()/std_err

        ddf = parameters.get("nobs") - constrained.shape[0]
        import scipy.stats as stats

        p_value = 2*(1-stats.t.cdf(np.abs(t_student),ddf))

        parameter_results = pd.DataFrame(np.vstack([constrained.squeeze(),
                                                    std_err,t_student,p_value])).T.round(4)



        splited_results = np.vsplit(parameter_results,indices)

        beta_sections = np.vsplit(splited_results[2],k_regimes)

        y_param_name = ["y%s"%i for i in np.arange(parameters.get("neqs_y"))]

        beta_sections_names = ["beta%s"%i for i in np.arange(parameters.get("neqs_x"))]

        d_ns = []
        # delta_sections = np.vsplit(splited_results[4], k_regimes)

        delta_sections = splited_results[4]

        if len(splited_results[4])>0:

            delta_sections_names = ["delta%s"%i for i in np.arange(parameters.get("neqs_z"))]

            d_ns = ["%s.%s"%(y_n,b_n) for y_n in y_param_name for b_n in delta_sections_names]

        b_ns = ["%s.%s"%(y_n,b_n) for b_n in beta_sections_names for y_n in y_param_name]

        parameters["y_param_names"] = y_param_name

        import time
        time_now = time.localtime()
        time_of_day = [time.strftime("%H:%M:%S", time_now)]
        date = time.strftime("%a, %d %b %Y", time_now)

        # sample=["1","2"]

        gen_left = [
            ('Dep. Variable:', [y_param_name]),
            ('Date:', [date]),
            ('Time:', [time_of_day]),
            # ('Sample:', [sample[0]]),
            # ('', [sample[1]]),
        ]

        gen_right = [
            ('No. Observations:', [parameters.get("nobs")]),
            ('Covariance Type', [parameters.get("covariance_type")]),
            ('Regimes',[parameters.get("k_regimes")]),
            ('Log Likelihood', ["%#5.3f" % results.get("likelihoods")]),

        ]

        gen_title = "Markov Switching Multivariate Model Results"


        if len(gen_right) < len(gen_left):
            # fill up with blank lines to same length
            gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
        elif len(gen_right) > len(gen_left):
            # fill up with blank lines to same length, just to keep it symmetric
            gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))

        gen_right = [('%-21s' % ('  ' + k), v) for k, v in gen_right]

        # gen_right = gen_left

        stubs = []
        vals = []
        for stub, val in gen_right:
            stubs.append(stub)
            vals.append(val)
        table_right = SimpleTable(vals, txt_fmt=fmt_2cols,  stubs=stubs)

        stubs = []
        vals = []
        for stub, val in gen_left:
            stubs.append(stub)
            vals.append(val)
        table_left = SimpleTable(vals, txt_fmt=fmt_2cols,title=gen_title, stubs=stubs)

        table_left.extend_right(table_right)

        summary = Summary()

        summary.tables.append(table_left)

        fmt2 = dict(
            data_fmts=["%s", "%s", "%s", "%s", "%s", "%s"],
            colwidths=15,
        )
        from copy import deepcopy
        _fmt_params = deepcopy(fmt_params)
        _fmt_params.update(fmt2)

        for i in np.arange(k_regimes):
            title = "Regime %s Switching Parameters"%i

            tbl = SimpleTable(data=beta_sections[i].values,
                        headers=["coef","std err","t","p>|t|"],stubs=b_ns,title=title,txt_fmt=_fmt_params)

            summary.tables.append(tbl)

        if self.model.has_delta:
            delta_tbl = SimpleTable(data=delta_sections.values,
                            headers=["coef","std err","t","p>|t|"],stubs=d_ns,title="Non Switching Parameters",txt_fmt=_fmt_params)

            summary.tables.append(delta_tbl)

        # p_j, p_ij, b, d, var_mat, inv_var_mat,det_inv_var_mat =\
        #     Markov_Multivarite_Regression.convert_param(self.model,constrained)
        p_j, p_ij, b, d, var_mat, inv_var_mat,det_inv_var_mat \
            = self.model.convert_param(constrained)
        # trans_df = pd.DataFrame(p_ij,index=np.arange(k_regimes),columns=np.arange(k_regimes))

        regime_parameter = parameter_results.iloc[0:indices[1]]

        # trans_information = pd.Series({"%s-%s"%(i,j): "%.4f"%c for i,r in enumerate(p_ij)for j,c in enumerate(r)})
        trans_stubs = ["p[%s-%s]"%(i,j) for i,r in enumerate(p_ij)for j,c in enumerate(r)][:indices[1]]

        regime_title = "Regime transition parameters"

        regime_tables = SimpleTable(data=regime_parameter.values,
                          headers=["coef1", "std err", "t", "p>|t|"], stubs=trans_stubs, title=regime_title,
                          txt_fmt=_fmt_params)

        summary.tables.append(regime_tables)

        parameter_results.columns = ["Estimates","Standard-errors","T","P-values"]

        return summary,parameter_results,p_ij,var_mat









