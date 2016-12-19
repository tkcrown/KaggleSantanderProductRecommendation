# -*- coding: utf-8 -*-
"""
Copy and tweak the best kernel

Code based on BreakfastPirate Forum post and forked from SRK
__author__ : MHK
"""
import csv
import datetime
import random
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble
from sklearn.model_selection import StratifiedKFold

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
target_cols = target_cols[2:]

missing_values = ['', 'NA']


def get_cols_vales(row, cols):
    def get_val(col):
        if row[col].strip() in missing_values:
            return 0
        else:
            return int(float(row[col]))

    return [get_val(col) for col in cols]


# the dataset is being ridiculous some how
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 2.00   24.00   39.00   40.19   50.00  164.00
def get_age(row):
    mean_age = 40.
    min_age = 18.
    max_age = 100.
    range_age = max_age - min_age
    age = row['age'].strip()
    if age in missing_values:
        age = mean_age
    else:
        age = float(age)
        if age < min_age:
            age = min_age
        elif age > max_age:
            age = max_age
    return round((age - min_age) / range_age, 4)


def get_cust_seniority(row):
    min_value = 0.
    max_value = 256.
    range_value = max_value - min_value
    cust_seniority = row['antiguedad'].strip()
    cust_seniority = float(cust_seniority)
    if cust_seniority < min_value:
        cust_seniority = min_value
    elif cust_seniority > max_value:
        cust_seniority = max_value
    return round((cust_seniority - min_value) / range_value, 4)


def get_income(row):
    min_value = 0.
    max_value = 1500000.
    range_value = max_value - min_value

    rent = row['renta'].strip()
    rent = float(rent)
    if rent < min_value:
        rent = min_value
    elif rent > max_value:
        rent = max_value

    return round((rent - min_value) / range_value, 6)


def get_marriage_Index(age, sex, income):
    # is this useful????
    marriage_age = 28
    modifier = 0
    if sex == 'V':
        modifier += -2
    if income <= 101850:
        modifier += -1

    marriage_age_mod = marriage_age + modifier

    if age <= marriage_age_mod:
        return 0
    else:
        return 1


def get_fecha_dato_month(row):
    return int(row['fecha_dato'].split('-')[1])


def process_data_mk(data_file, targets_dict, temporal_features_dict):
    x_vars_list = []
    y_vars_list = []
    instance_weights = []
    useful_months = [1, 2, 3, 4, 5, 6]

    reader = csv.DictReader(data_file)

    categ_cols = list(filter(lambda el: el.startswith("canal") or el.startswith("pais") or el.startswith("nomprov") or
                                        el.startswith("indrel_1mes") or el.startswith("ind_empleado"),
                             reader.fieldnames))
    temporal_cols = list(filter(lambda el: el.startswith("ind_actividad_cliente") or el.startswith("segmento") or
                                           el.startswith("tiprel_1mes"),
                             reader.fieldnames))

    print(categ_cols)
    print(temporal_cols)
    for row in reader:
        fecha_dato_month = get_fecha_dato_month(row)

        if fecha_dato_month not in useful_months:
            continue

        # Leave out first month
        customer_id = int(row['ncodpers'])

        if fecha_dato_month < 6:
            targets = get_cols_vales(row, target_cols)
            temporal_features = get_cols_vales(row, temporal_cols)
            targets_dict[fecha_dato_month][customer_id] = targets[:]
            temporal_features_dict[fecha_dato_month][customer_id] = temporal_features[:]
            continue

        # Only keep data for JUNE
        x_vars = []

        for col in categ_cols:
            x_vars.append(int(row[col]))

        ori_cols = ['sexo', 'ind_nuevo', 'indrel', 'indresi', 'indext',
                    'indfall', "fecha_alta_int"]
        for col in ori_cols:
            x_vars.append(int(row[col]))

        sex = row['sexo']
        age = get_age(row)
        x_vars.append(age)
        x_vars.append(get_fecha_dato_month(row))
        x_vars.append(get_cust_seniority(row))
        income = get_income(row)
        x_vars.append(income)
        x_vars.append(get_marriage_Index(age, sex, income))

        final_x_vars = x_vars
        for i in range(1, 6):
            final_x_vars += targets_dict[i].get(customer_id, [0] * len(target_cols))

        for i in range(1, 6):
            final_x_vars += temporal_features_dict[i].get(customer_id, [0] * len(temporal_cols))

        if row['fecha_dato'] == '2016-06-28':
            x_vars_list.append(final_x_vars)
        elif row['fecha_dato'] == '2015-06-28':
            targets = get_cols_vales(row, target_cols)
            targets_prev_month = targets_dict[5].get(customer_id, [0] * len(target_cols))
            assert len(targets_prev_month) == 22

            new_products = [target > target_prev_month for (target, target_prev_month) in
                            zip(targets, targets_prev_month)]
            n_new_products = sum(new_products)

            if n_new_products == 0:
                continue

            instance_weights.extend([1/n_new_products] * n_new_products)
            for i, new_product in enumerate(new_products):
                if new_product:
                    x_vars_list.append(final_x_vars)
                    y_vars_list.append(i)

    return x_vars_list, y_vars_list, targets_dict, temporal_features_dict, np.array(instance_weights)


def cross_validate_XGB(X, y, weights, n_folds=16, seed_val=0):
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=218)
    kf = kfold.split(X, y)
    xgb_hyper_params = {'objective': 'multi:softprob',
                        'eta': 0.1,
                        'max_depth': 5,
                        'silent': 1,
                        'num_class': 22,
                        'eval_metric': "mlogloss",
                        'min_child_weight': 5,
                        'subsample': 0.9,
                        'colsample_bytree': 0.8,
                        'seed': seed_val}

    num_rounds = 10000
    for i, (train_fold, validate) in enumerate(kf):
        if i == 2:
            # return xgb_hyper_params, int(128 * (1. + 1. / n_folds))
            X_train, X_validate, label_train, label_validate, weights_train, weights_valid = \
                X[train_fold, :], X[validate, :], y[train_fold], y[validate], \
                weights[train_fold], weights[validate]
            train_set = xgb.DMatrix(X_train, label_train, weight=weights_train)
            validation_set = xgb.DMatrix(X_validate, label_validate, weight=weights_valid)
            watchlist = [(train_set, 'train'), (validation_set, 'valid')]
            bst = xgb.train(xgb_hyper_params, train_set, num_rounds, evals=watchlist, verbose_eval=10,
                            early_stopping_rounds=50)
            best_trees = bst.best_ntree_limit
            print(best_trees)
            return xgb_hyper_params, int(best_trees * (1. + 1. / n_folds))


def run_xgb(train_X, train_y, hyper_params, num_rounds, weights):
    xgtrain = xgb.DMatrix(train_X, label=train_y, weight=weights)
    watchlist = [(xgtrain, 'train')]
    return xgb.train(hyper_params, xgtrain, num_rounds, evals=watchlist, verbose_eval=10)


# CONFIGS:
data_path = "../data/"
k_folds = 16

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    # Read and process train file
    train_file = open(data_path + "onehot_train.csv")
    print('Starting train file processing')
    x_vars_list, y_vars_list, targets_dict, features_dict, instance_weights = \
        process_data_mk(train_file, {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, {1: {}, 2: {}, 3: {}, 4: {}, 5: {}})
    print('Finished train file processing')
    train_X = np.array(x_vars_list)
    train_y = np.array(y_vars_list)
    print(np.unique(train_y))
    del x_vars_list, y_vars_list
    train_file.close()
    print(train_X.shape, train_y.shape)
    print(datetime.datetime.now() - start_time)
    out_df = pd.DataFrame(train_X)
    out_df.to_csv('trainx.csv', index=False)
    out_df = pd.DataFrame(train_y)
    out_df.to_csv('trainy.csv', index=False)

    # Read and process test file
    test_file = open(data_path + "onehot_test.csv")
    x_vars_list, y_vars_list, targets_dict, features_dict, dummy = process_data_mk(test_file, targets_dict, features_dict)
    test_X = np.array(x_vars_list)
    del x_vars_list
    test_file.close()
    print(test_X.shape)
    print(datetime.datetime.now() - start_time)

    # Cross validation and train
    print("CV to find best number of trees...")
    best_hyper_params, best_num_rounds = cross_validate_XGB(train_X, train_y, instance_weights, k_folds)
    print("Building model..")
    model = run_xgb(train_X, train_y, best_hyper_params, best_num_rounds, instance_weights)
    del train_X, train_y

    # Prediction
    print("Predicting..")
    xgtest = xgb.DMatrix(test_X)
    preds = model.predict(xgtest)
    print(preds)
    del test_X, xgtest
    print(datetime.datetime.now() - start_time)

    print("Getting the top products..")
    test_id = np.array(pd.read_csv(data_path + "onehot_test.csv", usecols=['ncodpers'])['ncodpers'])
    new_products = []
    for i, customer_id in enumerate(test_id):
        new_products.append([x1 - x2 for (x1, x2) in zip(preds[i, :], targets_dict[5][customer_id])])
    target_cols = np.array(target_cols)
    preds = np.argsort(np.array(new_products), axis=1)
    preds = np.fliplr(preds)[:, :7]
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv('sub_xgb_benchmark_with_R_onehot_with_weights.csv', index=False)
    print(datetime.datetime.now() - start_time)

