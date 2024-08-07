# %matplotlib inline
# %config Completer.use_jedi = False # to use autocomplete
import numpy as np
import os
import json
from json import JSONEncoder
import pandas as pd
import sklearn
from random import seed
from sklearn.model_selection import KFold
from sklearn import linear_model, ensemble, neighbors, tree
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
import scipy
import scipy.optimize as optimization
RANDOM_SEED=0
seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(sklearn.__version__)
from itertools import combinations
from numpy.random import default_rng
import itertools
import sys
import pickle

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, scipy.sparse.csr_matrix):
            try:
                tmp = obj.toarray().to_list()
            except:
                tmp = None
            return tmp
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, scipy.optimize.LbfgsInvHessProduct):
            return None
        return JSONEncoder.default(self, obj)

def path_leaf(path): # get filename(with extension) from path
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)


def file_in_folder(folder_path, flag=1): # get files in given folder, return list of filepath and filename
    file_list = []
    file_name = []
    for(dirpath, dirnames, filenames) in os.walk(folder_path):
#         print(filenames)
        
        for i in filenames:
            try:
                file_list += [dirpath + os.sep + i]
                file_name += [i]
            except:
                continue
        if flag == 0:
            break
        file_list.sort(key=path_leaf)
        file_name.sort()
    return [file_list, file_name]


class EU_model():
    def __init__(self, minimize_method=None, bound=None, model_type='e', initial_values=None, combine_average=True):
        # if True, combine all error then take average; if False, take average of domains then take average over three domains
        self.combine_average = combine_average
        
        self.fitted_parameters = None
        self.cov = None
        self.parameter_names = 'e'
        self.possible_params = ['e']
        if bound is None:
            self.param_bounds = {'e': (1e-6,100)}
        else:
            self.param_bounds = bound
        self.cur_min = 1000000
        self.minimize_method = minimize_method
        self.bound = tuple(self.param_bounds[i] for i in model_type)
        self.model_type=model_type
        if len(self.bound) != len(self.model_type):
            print('bound size is not matched to model type')
        if initial_values is None:
            self.initial_values = [1 for i in range(len(self.model_type))]
        else:
            self.initial_values = initial_values[:len(self.model_type)]
        pass
    
    
    def reset(self):
        self.fitted_parameters = None
        self.cov = None
        self.cur_min = 1000000
    
    def func(self, data, e):
        # data: [z1, z2, p1]
        if len(data.shape) == 1: 
            # only one example
            data = data.reshape(1, -1)
        z1 = data[:, 0]
        z2 = data[:, 1]
        p = data[:, 2]

        z1 = np.array(z1)
        z2 = np.array(z2)

        final_ce = []
        # print(z1.shape)
        for i in range(z1.shape[0]):
            c_z1 = z1[i]
            c_z2 = z2[i]
            c_p = p[i]
            # print(c_z1.shape, c_z2.shape)
            if c_z1 < 0:
                sign = -1
                c_z1 = -c_z1
                c_z2 = -c_z2
            else:
                sign = 1
            if np.abs(e-1) < 1e-6:
                # vgg = c_p*np.log(c_z1) + (1-c_p)*np.log(c_z2)
                final_ce.append(sign * (np.power(c_z1, c_p) * np.power(c_z2, 1-c_p)))
            else:
                # vgg = c_p*(np.power(c_z1, 1-e)-1)/(1-e) + (1-c_p)*(np.power(c_z2, 1-e)-1)/(1-e)
                # final_ce.append(sign * np.power(1+(1-e)*vgg, 1/(1-e)))
                # print(c_z1, c_z2, c_p, e, 1-e)
                if c_z2 == 0:
                    final_ce.append(sign * c_z1 * np.power(c_p+(1-c_p)/np.power(c_z1, 1-e), 1/(1-e)))
                else:
                    # print(c_p, c_z1, c_z2, 1-e, c_p*np.power(c_z1, 1-e)+(1-c_p)*np.power(c_z2, 1-e))
                    # final_ce.append(sign * np.power(c_p*np.power(c_z1, 1-e)+(1-c_p)*np.power(c_z2, 1-e), 1/(1-e)))
                    final_ce.append(sign * c_z1 * np.power(c_p + (1-c_p)*np.power(c_z2/c_z1, 1-e), 1/(1-e)))
        # print(len(final_ce), len(final_ce[0]))
        return np.array(final_ce)
    
    def complexity_func(self, to_optimize, X, y):
        # to_optimize = [alpha, beta, delta, gamma]
#         alpha, beta, delta, gamma = to_optimize
        a = to_optimize[0]

        y_pred = self.func(X, a)
        # print(y.shape, y_pred.shape)
        errors = metrics.mean_squared_error(y_true=y, y_pred=y_pred)
        inner_prod = errors
        if inner_prod < self.cur_min:
            # print(-inner_prod, alpha, beta, delta, gamma)
            self.cur_min = inner_prod
        return inner_prod 
    
        
    def cal_rademacher_complexity(self, X, y, sigma):
        res = optimization.minimize(self.complexity_func, self.initial_values, args=(X, y, sigma), method=self.minimize_method, bounds=self.bound)
        return res
    
    def update_param(self, params):
        tmp = []
        count = 0
        for i in self.possible_params:
            if i in self.model_type:
                tmp.append(params[count])
                count += 1
            else:
                tmp.append(None)
        self.fitted_parameters = tmp


    def fit(self, X, y, warm_start=None):
        res = optimization.minimize(self.complexity_func, self.initial_values, args=(X, y), method=self.minimize_method, bounds=self.bound)
        params = res['x']
        self.update_param(params)
        return res
    
    def predict(self, X):
        pred = self.func(X, *self.fitted_parameters)
        return pred

    def get_error(self, X, y, metric_type='mse'):
        pred = self.predict(X)
        if np.any(np.isnan(pred)):
            print('pred includes nan')
        if not np.all(np.isfinite(pred)):
            print('pred includes infinite')
            # print(X[np.array(np.isfinite(pred))*-1+1])
        if metric_type == 'mse':

            return metrics.mean_squared_error(y_true=y, y_pred=pred)
        else:
            print('ERROR: wrong metric type')
    
    def param_print(self):
        print(f'Parameters: {self.parameter_names}. {self.fitted_parameters}')





# get data

df = pd.read_csv('../../data/PPP_normalized_pooled_data/30countries.csv')

a, b = np.unique(df['lottery'], return_counts=True)

useful_lottery_num = []
for i in range(b.shape[0]):
    if b[i] > 2900:
        useful_lottery_num.append(int(a[i]))
print(useful_lottery_num, len(useful_lottery_num))
# in total 24 lotteries
num_domains = len(useful_lottery_num)

train_domain_num_batch = int(sys.argv[1])
# train_domain_num = train_domain_num
# experiment_type = int(sys.argv[2])

for train_domain_num in range(10*train_domain_num_batch, 10*(train_domain_num_batch+1)):
    if train_domain_num > 42503:
        continue
# training_combs = list(itertools.combinations(list(range(num_domains)), 1))
    training_combs = list(itertools.combinations(useful_lottery_num, 5))

    cur_comb = training_combs[train_domain_num]
    experiment_type = '_'.join([str(i) for i in cur_comb])
    train_domain = []

    for i in useful_lottery_num:
    # for i in range(2, 5):
        if i in cur_comb:
            train_domain.append(i)

    print(cur_comb, train_domain)

    # pooled data
    pooled_data = None
    training_cols = ['z1', 'z2', 'p1']
    target_col = ['ce']
    all_cols = training_cols + target_col
    data_sizes = []
    test_data = {}
    for lottery_num in useful_lottery_num:
        val = df[df['lottery'] == lottery_num]
        if lottery_num not in train_domain:
            test_data[lottery_num] = {'X': val[training_cols].values, 'y': val[target_col].values.reshape(-1,)}
            print(test_data[lottery_num]['X'].shape, test_data[lottery_num]['y'].shape)
        else:
            if pooled_data is None:
                pooled_data = val[all_cols]
            else:
                pooled_data = pd.concat((pooled_data, val[all_cols]))



    X = pooled_data[training_cols].values
    y = pooled_data[target_col].values.reshape(-1,)
    print(test_data.keys())
    print(X.shape, y.shape)

    print(test_data.keys())
    method_list = [
        # 'Nelder-Mead', 
        'Powell', 
        'L-BFGS-B', 
        'TNC', 
        # 'SLSQP', 
        'trust-constr'
    ]

    # seed_list = [0, 1, 2]
    # initial_points = [[1 for i in range(4)]]

    initial_points = [[1e-6 for i in range(4)], [0.5 for i in range(4)], [1 for i in range(4)]]
    initial_points = [[i] for i in range(1, 101, 20)]


    res_fol = 'in_sample'
    try:
        if not os.path.isdir(res_fol):
            os.mkdir(res_fol)
    except:
        pass
    res_fol = os.path.join(res_fol, f'{experiment_type}')
    try:
        if not os.path.isdir(res_fol):
            os.mkdir(res_fol)
    except:
        pass


    def check_range(model_type, values):
        param_bounds = {'e': (1e-6,100)}
        for idx, param in enumerate(param_bounds.keys()):
            left, right = param_bounds[param]
            if right is None:
                right = np.inf
            cur_value = values[idx]
            if cur_value < left or cur_value > right:
                return False
        return True

    res = {}
    count = 0
    # model_type = model_types[model_type_number]
    model_type = 'EU_iso'
    for method in method_list:
        # if count == cur_idx:
        # for model_type in model_types:

        print(model_type, train_domain_num, method)
        for num, initial in enumerate(initial_points):
            # cur_model = ML_models(model_type=model_type, random_seed=cur_seed)
            # cur_model.fit(X, y)

            # res_list, sigma_list = rademacher_complexity(model_type=i, minimize_method=method, X=X, y=y, num_sampling=1, bounds=None, iterate_all=False)
            
            # cur_model = CPT_model(minimize_method=method, model_type=model_type, initial_values=initial)
            cur_model = EU_model(minimize_method=method, initial_values=initial)
            res_list = cur_model.fit(X=X, y=y)
            # print(res_list)
            model_type = 'EU_iso'
            res['method'] = method
            res['model_type'] = model_type
            tmp = dict(res_list)
            res['initial'] = initial
            res['train_mse'] = tmp['fun']
            res['parameters'] = tmp['x']
            # res['train_domain'] = num_name_dic[train_domain_num]
            res['train_domain_num'] = train_domain_num
            success_flag = tmp['success']

            if not success_flag or not check_range(model_type, tmp['x']):
                print('not success or out of range')
                continue
            # print(res)
            res_file = os.path.join(res_fol, f'{model_type}.json')
            # res_model_file = os.path.join(res_fol, f'{model_type}.pkl')

            keep_old = False
            if os.path.isfile(res_file):
                print('compare with old result')
                with open(res_file, 'r') as f:
                    old_res = json.load(f)
                if old_res['train_mse'] < res['train_mse']:
                    keep_old = True
            # print(res)


            if not keep_old:
                print('Write new')
                print('generate test error')
                res['test_mse'] = {}
                for key, val in test_data.items():
                    # print(key, num_name_dic[key])
                    res['test_mse'][key] = cur_model.get_error(val['X'], val['y'])
                # print(res)
                res = json.dumps(res, cls=NumpyArrayEncoder)
                res = json.loads(res)
                with open(res_file, 'w') as f:
                    json.dump(res, f)






