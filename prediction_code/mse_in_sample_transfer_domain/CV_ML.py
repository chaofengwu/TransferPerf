# %matplotlib inline
# %config Completer.use_jedi = False # to use autocomplete
import numpy as np
import os
import json
from json import JSONEncoder
import pandas as pd
import sklearn
from random import seed
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn import linear_model, ensemble, neighbors, tree, kernel_ridge
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


class KR(sklearn.base.BaseEstimator):
    def __init__(self, kernel='rbf', gamma=None):
        self.kernel = kernel
        self.gamma = gamma
    
    def get_params(self, deep=True):
        return {'gamma': self.gamma}

    def set_params(self, **params):
        if 'gamma' in params.keys():
            self.gamma = params['gamma']
        return self

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        pred = []
        for i in X:
            tmp = i.reshape(1,-1)
            w = metrics.pairwise.rbf_kernel(self.X, tmp, gamma=self.gamma).reshape(-1,)
            if np.sum(w) == 0:
                w = np.ones((self.X.shape[0]))
            model_ft = linear_model.LinearRegression()
            model_ft.fit(self.X, self.y, sample_weight=w)
            pred.append(model_ft.predict(tmp))
        pred = np.array(pred).reshape(-1,)
        return pred

class ML_models():
    def __init__(self, model_type='DT', random_seed=0, CV_param=False, X=None, y=None):
        self.model_type = model_type
        self.random_seed = random_seed
        self.model_ft = self.create_model()
        # print(self.model_ft.get_params())
        if CV_param:
            param_to_search = {
                'RF': {
                    'max_depth': [i*5 for i in range(1, 11)], 

                }, 
                'kernel_ridge_rbf': {
                    'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1/3, 1],
                }
            }
            num_iter = 1
            cur_params = param_to_search[self.model_type]
            for key, val in cur_params.items():
                num_iter *= len(val)
            # print(num_iter)
            model_ft_cv = RandomizedSearchCV(estimator=self.model_ft, param_distributions=cur_params, n_iter=num_iter, scoring='neg_mean_squared_error')
            model_ft_cv.fit(X, y)
            self.model_ft = model_ft_cv.best_estimator_
            # print(self.model_ft.get_params())
    

    def create_model(self):
        if self.model_type == 'DT':
            return tree.DecisionTreeRegressor(random_state=self.random_seed)
        elif self.model_type == 'RF':
            return ensemble.RandomForestRegressor(random_state=self.random_seed)
        elif self.model_type == 'NN':
            return MLPRegressor(random_state=self.random_seed, 
                hidden_layer_sizes=(100,), solver='lbfgs', max_iter=5000, learning_rate='adaptive', learning_rate_init=1e-3, activation='tanh')
        elif self.model_type == 'Lasso':
            return linear_model.Lasso(random_state=self.random_seed)
        elif self.model_type == 'kernel_ridge_poly':
            return kernel_ridge.KernelRidge(degree=1, kernel="poly")
        elif self.model_type == 'kernel_ridge_rbf':
            # return KR()
            return kernel_ridge.KernelRidge(degree=1, kernel="rbf")
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

data_folder = '../../data/PPP_normalized_44'
file_list, file_name = file_in_folder(data_folder)
file_list = [i for i in file_list if '.csv' in i]
file_name = [i for i in file_name if '.csv' in i]
num_name_dic = {i: name for i, name in enumerate(file_name)}
name_num_dic = {name: i for i, name in enumerate(file_name)}

# print(num_name_dic)
data_dic = {}
for idx, file in enumerate(file_list):
    df = pd.read_csv(file)
#     display(df)
    data_dic[file_name[idx]] = df

num_domains = len(num_name_dic.keys())

# total 44
train_domain_num = int(sys.argv[1])
cur_idx = train_domain_num
# experiment_type = int(sys.argv[2])

training_combs = list(itertools.combinations(list(range(num_domains)), 1))
# training_combs = list(itertools.combinations(list(range(16)), 6))
cur_comb = training_combs[train_domain_num]
experiment_type = '_'.join([str(i) for i in cur_comb])
train_domain = []
# test_domain = []
for i in range(num_domains):
# for i in range(2, 5):
    if i in cur_comb:
        train_domain.append(i)
        # test_domain.append(i)
    # else:
    #     test_domain.append(i)

print(train_domain)

# pooled data
pooled_data = None
training_cols = ['z1', 'z2', 'p1']
target_col = ['ce']
all_cols = training_cols + target_col
data_sizes = []
test_data = {}
for name_key, val in data_dic.items():
    num_key = name_num_dic[name_key]

    if num_key not in train_domain:
        test_data[num_key] = {'X': val[training_cols].values, 'y': val[target_col].values.reshape(-1,)}
        print(test_data[num_key]['X'].shape, test_data[num_key]['y'].shape)
    else:

        if pooled_data is None:
            pooled_data = val[all_cols]
        else:
            pooled_data = pd.concat((pooled_data, val[all_cols]))


X = pooled_data[training_cols].values
y = pooled_data[target_col].values.reshape(-1,)


seed_list = [0]


model_types = ['DT', 'RF', 'NN']
model_types = ['Lasso']
model_types = ['kernel_ridge_poly', 'kernel_ridge_rbf']
model_types = ['RF', 'kernel_ridge_rbf']
# model_types = ['kernel_ridge_rbf']

print(model_types)


res_fol = 'in_sample'
# res_fol = 'lasso test'
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


# print(X, y)
res = {}
count = 0
for model_type in model_types:
    for cur_seed in seed_list:
        # if count == cur_idx:
        print(cur_idx, model_type, cur_seed, experiment_type)
        cur_model = ML_models(model_type=model_type, random_seed=cur_seed, CV_param=True, X=X, y=y)
        cur_model.fit(X, y)
        error = cur_model.get_error(X, y)
        res['seed'] = cur_seed
        res['model_type'] = model_type
        res['train_mse'] = error
        res['train_domain'] = num_name_dic[train_domain_num]
        res['train_domain_num'] = train_domain_num

        res_file = os.path.join(res_fol, f'{model_type}_new.json')
        # res_model_file = os.path.join(res_fol, f'{model_type}.pkl')
        keep_old = False
        if os.path.isfile(res_file):
            print('compare with old result')
            with open(res_file, 'r') as f:
                old_res = json.load(f)
            if old_res['train_mse'] < res['train_mse']:
                keep_old = True
        
        if not keep_old:
            print('Write new')
            print('generate test error')
            res['test_mse'] = {}
            for key, val in test_data.items():
                res['test_mse'][key] = cur_model.get_error(val['X'], val['y'])
  
            with open(res_file, 'w') as f:
                json.dump(res, f)

        count += 1




