'''
xgboost submission
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

train_path = "../data/train201506_v0.csv"
test_path = "../data/test201506_v0.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

original_labels = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                   'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                   'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                   'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                   'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                   'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                   'ind_nomina_ult1',   'ind_nom_pens_ult1', 'ind_recibo_ult1']

label_mapper = {i:original_labels[i] for i in xrange(len(original_labels))}
label_rev_mapper = {original_labels[i]: i for i in xrange(len(original_labels))}


def transform_label(col):
    return np.array([label_rev_mapper[i] for i in col])


num_class = len(original_labels)

train_label = transform_label(train['product_id'])

train_id = train['ncodpers']

train.drop(['product_id','ncodpers'], axis=1, inplace=True)

train = train.as_matrix()

test_id = test['ncodpers']

test.drop(['ncodpers'], axis = 1, inplace=True)

test = test.as_matrix()

label_mapper = {i:original_labels[i] for i in xrange(len(original_labels))}


# xgboost
k = 1
eta = 0.1
colsample_bytree = 1.
max_depth = 10
min_child_weight = 50
subsample = 1.
num_boost_round = 100


final_pred = np.zeros((len(test_id), num_class))

for s in xrange(k):
    print s
    params = {"objective": "multi:softprob",
              "num_class": num_class,
              "booster": "gbtree",
              "eval_metric": "mlogloss",
              "eta": eta,
              "max_depth": int(max_depth),
              "subsample": subsample,
              "colsample_bytree": colsample_bytree,
              #"gamma": gamma,
              #"lamb": lamb,
              #"alpha": alpha,
              "min_child_weight": min_child_weight,
              #"colsample_bylevel": colsample_bylevel,
              "silent": 1,
              "seed": s
              }

    dtrain = xgb.DMatrix(train, train_label)
    dtest = xgb.DMatrix(test)
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=10)
    final_pred += bst.predict(dtest)


def get_top7(np_row):
    top7 = np.argsort(np_row)[::-1][:7]
    return ' '.join([label_mapper[i] for i in top7])

result = np.apply_along_axis(get_top7, 1, final_pred)

pd.DataFrame({'ncodpers': test_id, 'added_products': result}).to_csv("../data/xgb_pred_1.csv", index=False)



