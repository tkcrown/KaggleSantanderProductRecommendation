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

categ_col_value_mapping_dict = {'ind_empleado': {-99: 0, 'N': 1, 'B': 2, 'F': 3, 'A': 4, 'S': 5},
                                'sexo': {'V': 0, 'H': 1, -99: 2},
                                'ind_nuevo': {'0': 0, '1': 1, -99: 1},
                                'indrel': {'1': 0, '99': 1, -99: 1},
                                'indrel_1mes': {-99: 0, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4,
                                                '4': 4, 'P': 5},
                                'tiprel_1mes': {-99: 0, 'I': 1, 'A': 2, 'P': 3, 'R': 4, 'N': 5},
                                'indresi': {-99: 0, 'S': 1, 'N': 2},
                                'indext': {-99: 0, 'S': 1, 'N': 2},
                                # 'conyuemp'      : {-99:0, 'S':1, 'N':2},
                                'indfall': {-99: 0, 'S': 1, 'N': 2},
                                # 'tipodom'       : {-99:0, '1':1},
                                'ind_actividad_cliente': {'0': 0, '1': 1, -99: 2},
                                'segmento': {'02 - PARTICULARES': 0, '03 - UNIVERSITARIO': 1, '01 - TOP': 2, -99: 3},
                                'pais_residencia': {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62,
                                                    'JP': 82, 'JM': 116,
                                                    'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41,
                                                    'GW': 99, 'GT': 44,
                                                    'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98,
                                                    'GM': 110, 'GI': 96,
                                                    'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22,
                                                    'AD': 35, 'PR': 40,
                                                    'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91,
                                                    'PL': 30, 'EE': 52,
                                                    'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54,
                                                    'ZW': 114, 'ES': 0,
                                                    'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118,
                                                    'MR': 48, 'UA': 49,
                                                    'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7,
                                                    'NO': 46,
                                                    'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55,
                                                    'CL': 4,
                                                    'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32,
                                                    'CU': 72, 'KE': 65,
                                                    'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47,
                                                    'SL': 97, 'KZ': 111,
                                                    'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76,
                                                    'DE': 10, 'DZ': 80,
                                                    'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85,
                                                    'LT': 103, 'LU': 59,
                                                    'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107,
                                                    'IT': 18, 'AO': 71,
                                                    'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
                                'canal_entrada': {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10,
                                                  'KHL': 0,
                                                  'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2,
                                                  'KHE': 1, 'KHF': 19,
                                                  '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23,
                                                  'KAE': 30,
                                                  'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94,
                                                  'KAN': 110,
                                                  'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46,
                                                  'KAW': 76,
                                                  'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133,
                                                  'KBH': 90,
                                                  'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102,
                                                  'KBG': 17,
                                                  'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101,
                                                  'KBS': 118,
                                                  'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86,
                                                  'KCD': 85,
                                                  'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82,
                                                  'KCL': 53,
                                                  'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156,
                                                  'KCU': 115,
                                                  'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153,
                                                  'KCX': 120,
                                                  'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14,
                                                  'KDI': 150,
                                                  'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117,
                                                  'KDC': 75,
                                                  'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91,
                                                  'KDW': 132,
                                                  'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96,
                                                  'KEN': 137,
                                                  'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15,
                                                  'KEG': 136,
                                                  'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89,
                                                  'KEZ': 108,
                                                  'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138,
                                                  -99: 6, 'KFV': 48,
                                                  'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45,
                                                  'KFG': 27,
                                                  'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42,
                                                  'KFL': 34,
                                                  'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71,
                                                  '004': 83,
                                                  'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18,
                                                  'KGN': 11}
                                }

categ_cols = list(categ_col_value_mapping_dict.keys())
categ_cols.sort()

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
target_cols = target_cols[2:]


def get_target_vals(row):
    def get_val(col):
        if row[col].strip() in ['', 'NA']:
            return 0
        else:
            return int(float(row[col]))

    return [get_val(col) for col in target_cols]


def get_index_for_categ_col(row, col):
    val = row[col].strip()
    if val in ['', 'NA']:
        val = -99
    return categ_col_value_mapping_dict[col][val]


def get_age(row):
    mean_age = 40.
    min_age = 18.
    max_age = 100.
    range_age = max_age - min_age
    age = row['age'].strip()
    if age == 'NA' or age == '':
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
    impute_val = 0.
    cust_seniority = row['antiguedad'].strip()
    if cust_seniority == 'NA' or cust_seniority == '':
        cust_seniority = impute_val
    else:
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
    # the region-specific ranta value
    renta_dict = {'ALBACETE': 76895, 'ALICANTE': 60562, 'ALMERIA': 77815, 'ASTURIAS': 83995, 'AVILA': 78525,
                  'BADAJOZ': 60155, 'BALEARS, ILLES': 114223, 'BARCELONA': 135149, 'BURGOS': 87410, 'NAVARRA': 101850,
                  'CACERES': 78691, 'CADIZ': 75397, 'CANTABRIA': 87142, 'CASTELLON': 70359, 'CEUTA': 333283,
                  'CIUDAD REAL': 61962, 'CORDOBA': 63260, 'CORUÑA, A': 103567, 'CUENCA': 70751, 'GIRONA': 100208,
                  'GRANADA': 80489,
                  'GUADALAJARA': 100635, 'HUELVA': 75534, 'HUESCA': 80324, 'JAEN': 67016, 'LEON': 76339,
                  'LERIDA': 59191, 'LUGO': 68219, 'MADRID': 141381, 'MALAGA': 89534, 'MELILLA': 116469,
                  'GIPUZKOA': 101850,
                  'MURCIA': 68713, 'OURENSE': 78776, 'PALENCIA': 90843, 'PALMAS, LAS': 78168, 'PONTEVEDRA': 94328,
                  'RIOJA, LA': 91545, 'SALAMANCA': 88738, 'SANTA CRUZ DE TENERIFE': 83383, 'ALAVA': 101850,
                  'BIZKAIA': 101850,
                  'SEGOVIA': 81287, 'SEVILLA': 94814, 'SORIA': 71615, 'TARRAGONA': 81330, 'TERUEL': 64053,
                  'TOLEDO': 65242, 'UNKNOWN': 103689, 'VALENCIA': 73463, 'VALLADOLID': 92032, 'ZAMORA': 73727,
                  'ZARAGOZA': 98827}

    # missing_value = 101850.
    rent = row['renta'].strip()
    if rent == 'NA' or rent == '':
        if row['nomprov'] == 'NA' or row['nomprov'] == '':
            region = 'UNKNOWN'
        else:
            region = row['nomprov']
        rent = float(renta_dict[region])
    else:
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


def get_fecha_alta_month(row):
    if row['fecha_alta'].strip() == 'NA' or row['fecha_alta'].strip() == '':
        return int(random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    else:
        return int(row['fecha_alta'].split('-')[1])


def process_data_mk(data_file, targets_dict_may, targets_dict_jan, targets_dict_feb, targets_dict_mar,
                    targets_dict_apr):
    x_vars_list = []
    y_vars_list = []
    useful_dates = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28', '2015-06-28',
                    '2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28', '2016-06-28']
    n = 0
    for row in csv.DictReader(data_file):
        if row['fecha_dato'] not in useful_dates:
            continue

        # Leave out first month
        customer_id = int(row['ncodpers'])

        if get_fecha_dato_month(row) != 6:
            targets = get_target_vals(row)

        if get_fecha_dato_month(row) == 1:
            targets_dict_jan[customer_id] = targets[:]
            continue

        if get_fecha_dato_month(row) == 2:
            targets_dict_feb[customer_id] = targets[:]
            continue

        if get_fecha_dato_month(row) == 3:
            targets_dict_mar[customer_id] = targets[:]
            continue

        if get_fecha_dato_month(row) == 4:
            targets_dict_apr[customer_id] = targets[:]
            continue

        if get_fecha_dato_month(row) == 5:
            targets_dict_may[customer_id] = targets[:]
            continue

        # Only keep data for JUNE
        x_vars = []
        for col in categ_cols:
            x_vars.append(get_index_for_categ_col(row, col))

        sex = get_index_for_categ_col(row, 'sexo')
        # gender was not included in previous version
        # x_vars.append(sex)
        age = get_age(row)
        x_vars.append(age)
        x_vars.append(get_fecha_dato_month(row))
        x_vars.append(get_fecha_alta_month(row))
        x_vars.append(get_cust_seniority(row))
        income = get_income(row)
        x_vars.append(income)
        x_vars.append(get_marriage_Index(age, sex, income))

        targets_prev_month = targets_dict_may.get(customer_id, [0] * 22)
        lag_target_list_jan = targets_dict_jan.get(customer_id, [0] * 22)
        lag_target_list_feb = targets_dict_feb.get(customer_id, [0] * 22)
        lag_target_list_mar = targets_dict_mar.get(customer_id, [0] * 22)
        lag_target_list_apr = targets_dict_apr.get(customer_id, [0] * 22)
        assert len(targets_prev_month) == 22
        final_x_vars = x_vars + targets_prev_month + lag_target_list_jan + lag_target_list_feb + lag_target_list_mar +\
                       lag_target_list_apr

        if row['fecha_dato'] == '2016-06-28':
            x_vars_list.append(final_x_vars)
            n+=1
        elif row['fecha_dato'] == '2015-06-28':
            targets = get_target_vals(row)
            new_products = [target > target_prev_month for (target, target_prev_month) in
                            zip(targets, targets_prev_month)]

            for i, new_product in enumerate(new_products):
                if new_product:
                    x_vars_list.append(final_x_vars)
                    y_vars_list.append(i)
                    n+=1

    return x_vars_list, y_vars_list, targets_dict_may, targets_dict_jan, targets_dict_feb, targets_dict_mar, targets_dict_apr


def cross_validate_XGB(X, y, n_folds=16, seed_val=0):
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
            return xgb_hyper_params, int(128 * (1. + 1. / n_folds))
            X_train, X_validate, label_train, label_validate = \
                X[train_fold, :], X[validate, :], y[train_fold], y[validate]
            train_set = xgb.DMatrix(X_train, label_train)
            validation_set = xgb.DMatrix(X_validate, label_validate)
            watchlist = [(train_set, 'train'), (validation_set, 'valid')]
            bst = xgb.train(xgb_hyper_params, train_set, num_rounds, evals=watchlist, verbose_eval=10,
                            early_stopping_rounds=50)
            best_trees = bst.best_ntree_limit
            print(best_trees)
            return xgb_hyper_params, int(best_trees * (1. + 1. / n_folds))


def run_xgb(train_X, train_y, hyper_params, num_rounds):
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    watchlist = [(xgtrain, 'train')]
    return xgb.train(hyper_params, xgtrain, num_rounds, evals=watchlist, verbose_eval=10)


# CONFIGS:
data_path = "../input/"
k_folds = 16

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    # Read and process train file
    train_file = open(data_path + "train_ver2.csv")
    print('Starting train file processing')
    # x_vars_list, y_vars_list, cust_dict = processData(train_file, {})
    x_vars_list, y_vars_list, targets_dict_may, targets_dict_jan, targets_dict_feb, targets_dict_mar, targets_dict_apr = \
        process_data_mk(train_file, {}, {}, {}, {}, {})
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
    test_file = open(data_path + "test_ver2.csv")
    x_vars_list, y_vars_list, targets_dict_may, targets_dict_jan, targets_dict_feb, targets_dict_mar, targets_dict_apr = \
        process_data_mk(test_file, targets_dict_may, targets_dict_jan, targets_dict_feb, targets_dict_mar,
                        targets_dict_apr)
    test_X = np.array(x_vars_list)
    del x_vars_list
    test_file.close()
    print(test_X.shape)
    print(datetime.datetime.now() - start_time)

    print("CV to find best number of trees...")
    best_hyper_params, best_num_rounds = cross_validate_XGB(train_X, train_y, k_folds)
    print("Building model..")
    model = run_xgb(train_X, train_y, best_hyper_params, best_num_rounds)
    del train_X, train_y
    print("Predicting..")
    xgtest = xgb.DMatrix(test_X)
    preds = model.predict(xgtest)
    print(preds)
    del test_X, xgtest
    print(datetime.datetime.now() - start_time)

    print("Getting the top products..")
    test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
    new_products = []
    for i, customer_id in enumerate(test_id):
        new_products.append([x1 - x2 for (x1, x2) in zip(preds[i, :], targets_dict_may[customer_id])])
    target_cols = np.array(target_cols)
    preds = np.argsort(np.array(new_products), axis=1)
    preds = np.fliplr(preds)[:, :7]
    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
    out_df = pd.DataFrame({'ncodpers': test_id, 'added_products': final_preds})
    out_df.to_csv('sub_xgb_benchmark2.csv', index=False)
    print(datetime.datetime.now() - start_time)
