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


class CPT_model():
    def __init__(self, minimize_method=None, bound=None, model_type='abdg', initial_values=None, combine_average=True):
        # if True, combine all error then take average; if False, take average of domains then take average over three domains
        self.combine_average = combine_average
        
        self.fitted_parameters = None
        self.cov = None
        self.parameter_names = 'alpha, beta, delta, gamma'
        self.possible_params = ['a', 'b', 'd', 'g']
        if bound is None:
            self.param_bounds = {'a': (1e-8,1), 'b': (1e-8,1), 'd': (1e-8,None), 'g': (1e-8,1)}
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
    
    def func(self, data, alpha, beta, delta, gamma):
        # data: [z1, z2, p1]
        if len(data.shape) == 1: 
            # only one example
            data = data.reshape(1, -1)
        z1 = data[:, 0]
        z2 = data[:, 1]
        p = data[:, 2]
        if alpha is None:
            alpha = 1
        if beta is None:
            beta = 1
        if delta is None:
            delta = 1
        if gamma is None:
            gamma = 1
        wp = delta*np.power(p, gamma) / (delta*np.power(p, gamma) + np.power(1-p, gamma))

        vz1 = []
        vz2 = []

        for i in range(len(data)):
            if z1[i] >= 0:
                vz1.append(np.power(z1[i], alpha))
            else:
                vz1.append(-1*np.power(-1*z1[i], beta))
            if z2[i] >= 0:
                vz2.append(np.power(z2[i], alpha))
            else:
                vz2.append(-1*np.power(-1*z2[i], beta))

        vz1 = np.array(vz1)
        vz2 = np.array(vz2)

        vgg = wp*vz1 + (1-wp)*vz2

        final_ce = []
        for i in vgg:
            if i >= 0:
                final_ce.append(np.power(i, 1/alpha))
            else:
                final_ce.append(-1*np.power(-i, 1/beta))

        return np.array(final_ce)
    
    def complexity_func(self, to_optimize, X, y):
        # to_optimize = [alpha, beta, delta, gamma]
#         alpha, beta, delta, gamma = to_optimize
        alpha = None
        beta = None
        delta = None
        gamma = None
        visited_param = {}
        for i in to_optimize:
            if 'a' not in visited_param.keys() and 'a' in self.model_type:
                alpha = i
                visited_param['a'] = 1
                continue
            if 'b' not in visited_param.keys() and 'b' in self.model_type:
                beta = i
                visited_param['b'] = 1
                continue
            if 'd' not in visited_param.keys() and 'd' in self.model_type:
                delta = i
                visited_param['d'] = 1
                continue
            if 'g' not in visited_param.keys() and 'g' in self.model_type:
                gamma = i
                visited_param['g'] = 1
                continue


        # errors = []
        # for idx in range(len(X)):
        #     cur_X = X[idx]
        #     cur_y = y[idx]
        #     y_pred = self.func(cur_X, alpha, beta, delta, gamma)


        #     # cur_error = np.sum(np.power(cur_y-y_pred, 2)) / len(y_pred)
        #     cur_error = np.power(cur_y-y_pred, 2)
        #     if self.combine_average:
        #         if len(errors) == 0:
        #             errors.append(cur_error)
        #         else:
        #             errors[0] = np.concatenate((errors[0], cur_error))
        #     else:
        #         errors.append(np.mean(cur_error))
        #     # print(cur_y.shape, y_pred.shape, cur_error.shape)
        #     # print(cur_error.shape, y_pred.shape)
        
        # errors = np.array(errors)
        # errors = np.mean(errors)
        # # print(errors, errors.shape)
        # # print(errors, sigma)
        # # inner_prod = np.dot(errors, sigma) / len(errors)

        y_pred = self.func(X, alpha, beta, delta, gamma)
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


class ML_models():
    def __init__(self, model_type='DT', random_seed=0):
        self.model_type = model_type
        self.random_seed = random_seed
        self.model_ft = self.create_model()

    def create_model(self):
        if self.model_type == 'DT':
            return tree.DecisionTreeRegressor(random_state=self.random_seed)
        elif self.model_type == 'RF':
            return ensemble.RandomForestRegressor(random_state=self.random_seed)
        elif self.model_type == 'NN':
            return MLPRegressor(random_state=self.random_seed)
        else:
            print('ERROR: wrong model type')

    def fit(self, X, y):
        self.model_ft.fit(X, y)

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


    def predict(self, X):
        return self.model_ft.predict(X)

    def save_model(self, path=None):
        with open(path, 'wb') as f:
            pickle.dump(self.model_ft, f)


    def load_model(self, path=None):
        with open(path, 'rb') as f:
            self.model_ft = pickle.load(f)




# get data
data_folder = '../../../data/PPP_normalized_44'
file_list, file_name = file_in_folder(data_folder)
file_list = [i for i in file_list if '.csv' in i]
file_name = [i for i in file_name if '.csv' in i]
num_name_dic = {i: name for i, name in enumerate(file_name)}
name_num_dic = {name: i for i, name in enumerate(file_name)}

print(num_name_dic)
data_dic = {}
for idx, file in enumerate(file_list):
    df = pd.read_csv(file)
#     display(df)
    data_dic[file_name[idx]] = df


num_domains = len(num_name_dic.keys())


all_params = ['a', 'b', 'd', 'g']
model_types = []
for i in range(1, len(all_params)+1):
    tmp = list(itertools.combinations(all_params, i))
    model_types += tmp
    
model_types = [''.join(i) for i in model_types]
# model_types = [model_types[-1]]
# model_types = ['g', 'ab', 'dg', 'abg', 'abdg']
print(model_types)

# total: 44 * 15 = 660
input_num = int(sys.argv[1])
train_domain_num = input_num // len(model_types)
model_type_number = input_num % len(model_types)
# print(train_domain_num, model_type_number)

training_combs = list(itertools.combinations(list(range(num_domains)), 1))
# training_combs = list(itertools.combinations(list(range(16)), 6))
cur_comb = training_combs[train_domain_num]
experiment_type = '_'.join([str(i) for i in cur_comb])
train_domain = []

for i in range(num_domains):
    if i in cur_comb:
        train_domain.append(i)


print(train_domain)


# pooled data
pooled_data = None
training_cols = ['z1', 'z2', 'p1']
target_col = ['ce']
all_cols = training_cols + target_col
data_sizes = []
test_data = {}
for name_key, val in data_dic.items():
    # print(name_key, val.shape)
    num_key = name_num_dic[name_key]

    if num_key not in train_domain:
        test_data[num_key] = {'X': val[training_cols].values, 'y': val[target_col].values.reshape(-1,)}
        print(test_data[num_key]['X'].shape, test_data[num_key]['y'].shape)
    else:
    # data_sizes.append(val.shape[0])
        if pooled_data is None:
            pooled_data = val[all_cols]
        else:
            pooled_data = pd.concat((pooled_data, val[all_cols]))


X = pooled_data[training_cols].values
y = pooled_data[target_col].values.reshape(-1,)

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
    param_bounds = {'a': (1e-8,1), 'b': (1e-8,1), 'd': (1e-8,None), 'g': (1e-8,1)}
    for idx, param in enumerate(model_type):
        left, right = param_bounds[param]
        if right is None:
            right = np.inf
        cur_value = values[idx]
        if cur_value < left or cur_value > right:
            return False
    return True

res = {}
count = 0
model_type = model_types[model_type_number]
for method in method_list:
    # if count == cur_idx:
    # for model_type in model_types:

    print(model_type, train_domain_num, method)
    for num, initial in enumerate(initial_points):
        # cur_model = ML_models(model_type=model_type, random_seed=cur_seed)
        # cur_model.fit(X, y)

        # res_list, sigma_list = rademacher_complexity(model_type=i, minimize_method=method, X=X, y=y, num_sampling=1, bounds=None, iterate_all=False)
        
        cur_model = CPT_model(minimize_method=method, model_type=model_type, initial_values=initial)
        res_list = cur_model.fit(X=X, y=y)
        print(res_list)
        res['method'] = method
        res['model_type'] = model_type
        tmp = dict(res_list)
        res['initial'] = initial
        res['train_mse'] = tmp['fun']
        res['parameters'] = tmp['x']
        res['train_domain'] = num_name_dic[train_domain_num]
        res['train_domain_num'] = train_domain_num
        success_flag = tmp['success']

        if not success_flag or not check_range(model_type, tmp['x']):
            print('not success or out of range')
            continue

        res_file = os.path.join(res_fol, f'{model_type}.json')
        # res_model_file = os.path.join(res_fol, f'{model_type}.pkl')

        keep_old = False
        if os.path.isfile(res_file):
            print('compare with old result')
            with open(res_file, 'r') as f:
                old_res = json.load(f)
            if old_res['train_mse'] < res['train_mse']:
                keep_old = True
        print(res)


        if not keep_old:
            print('Write new')
            print('generate test error')
            res['test_mse'] = {}
            for key, val in test_data.items():
                # print(key, num_name_dic[key])
                res['test_mse'][key] = cur_model.get_error(val['X'], val['y'])
            print(res)
            res = json.dumps(res, cls=NumpyArrayEncoder)
            res = json.loads(res)
            with open(res_file, 'w') as f:
                json.dump(res, f)






