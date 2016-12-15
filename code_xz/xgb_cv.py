'''
xgb cv
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from average_precision import mapk
from sklearn.preprocessing import LabelEncoder

def get_top7(np_row):
    return np.argsort(np_row)[::-1][:7]

train_path = "../data/train201506_v0.csv"

train = pd.read_csv(train_path)

original_labels = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                   'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                   'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                   'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                   'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                   'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                   'ind_nomina_ult1',   'ind_nom_pens_ult1', 'ind_recibo_ult1']

num_class = len(original_labels)

#label_mapper = {i:original_labels[i] for i in xrange(len(original_labels))}

#label_array = train[original_labels].as_matrix()

#def get_label(np_row):
#    return np.nonzero(np_row == 1)[0][0]

le = LabelEncoder()
train_label = le.fit_transform(train['product_id'])

train_id = train['ncodpers']

train.drop(['product_id','ncodpers'], axis=1, inplace=True)

train = train.as_matrix()

# xgboost
k = 1
NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state= 218)
eta = 0.1
colsample_bytree = 1.
max_depth = 10
min_child_weight = 50
subsample = 1.
num_boost_round = 5


cv_train = np.zeros((len(train_label), num_class))

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

    kf = kfold.split(train, train_label)

    for i, (train_fold, validate) in enumerate(kf):
        X_train, X_validate, label_train, label_validate = \
            train[train_fold, :], train[validate, :], train_label[train_fold], train_label[validate]

        dtrain = xgb.DMatrix(X_train, label_train)
        dvalid = xgb.DMatrix(X_validate, label_validate)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=10, early_stopping_rounds=50)
        cv_train[validate, :] += bst.predict(xgb.DMatrix(X_validate))
        tmp_result = list(np.apply_along_axis(get_top7, 1, cv_train[validate, :]))
        print mapk([[x] for x in label_validate],tmp_result)


result = list(np.apply_along_axis(get_top7, 1, cv_train))
print mapk([[x] for x in train_label], result)


