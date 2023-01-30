import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import xgboost as xgb
from cbfv.cbfv.composition import generate_features
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import uniform, randint

#   Helper function
def evaluate_model_hpsearch(model, model_name, X_train, y_train):
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    rmse = mean_squared_error(y_train, y_pred, squared=False)
    
    result_dict = {
        'model_name': model_name,
        'model_name_pretty': type(model).__name__,
        'model_params': model.get_params(),
        'r2': r2,
        'mae': mae,
        'rmse': rmse}
    return model, result_dict

def evaluate_model(model, model_name, X_train, y_train, X_val, y_val):
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    
    y_pred_val = model.predict(X_val)
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    
    result_dict = {
        #'Percentage of data': (1-percentage)*100,
        'model_name': model_name,
        'model_name_pretty': type(model).__name__,
        'model_params': model.get_params(),
        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_val': r2_val,
        'mae_val': mae_val,
        'rmse_val': rmse_val}
    return model, result_dict


  
# Creating a directory to save results.
directory_results = "results"
os.mkdir(directory_results)
save_path_results = "{}".format(directory_results)


regression_tasks = ['matbench_expt_gap', 'matbench_phonons', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form']

for task in regression_tasks:

    # Reading from csv, CBFV, scaling
    df = pd.read_csv('matbench_datasets/{}.csv'.format(task))

    #  Using CBFV and embedding vectors to featurize data
    X, y, formulae, skipped = generate_features(df, elem_prop='robo_descriptor', drop_duplicates=True)
    X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #   Scaling and normalizing
    scaler = StandardScaler()
    X_train = normalize(scaler.fit_transform(X_train_unscaled))
    X_test = normalize(scaler.transform(X_test_unscaled))

    #  CBFV mat2vec
    X_O, y_O, formulae_O, skipped_O = generate_features(df, elem_prop='mat2vec', drop_duplicates=True)

    #  CBFV onehot
    X_b, y_b, formulae_b, skipped_b = generate_features(df, elem_prop='onehot', drop_duplicates=True)
    


    #   Getting best model
    xgb_model = xgb.XGBRegressor(objective="reg:linear")

    params = {
        "max_depth": randint(4, 12), # default 3
        "learning_rate": uniform(0.03, 0.3), # default 0.1
        "colsample_bytree": uniform(0.7, 0.3),
        "min_child_weight": uniform(0.05, 0.2),
        "reg_lambda": uniform(0.7, 2.0),
        "n_estimators": randint(120, 170),
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=10, cv=10, verbose=1, n_jobs=7, return_train_score=True)
    search.fit(X_train, y_train)


    
    percentages = [p for p in range(10,100, 10)]
    per = [round((p/100), 1) for p in percentages]
    N_datapoints = []
    r2 = []
    mae = []
    rmse = []

    for p in per:
        X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X, y, train_size=p)

        #   Scaling and normalizing
        scaler = StandardScaler()

        X_train = normalize(scaler.fit_transform(X_train_unscaled))
        X_test = normalize(scaler.transform(X_test_unscaled))

        best_model = search.best_estimator_.fit(X_train, y_train)
        model, result_dict = evaluate_model(best_model, 'XGBoost', X_train, y_train, X_test, y_test)
        N_datapoints.append(len(X_train))
        r2.append(result_dict['r2_val'])
        mae.append(result_dict['mae_val'])
        rmse.append(result_dict['rmse_val'])


    #   base model

    r2_base = []
    mae_base = []
    rmse_base = []

    for p in per:
        X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X_O, y_O, train_size=p)

        #   Scaling and normalizing
        scaler = StandardScaler()

        X_train = normalize(scaler.fit_transform(X_train_unscaled))
        X_test = normalize(scaler.transform(X_test_unscaled))

        best_model = search.best_estimator_.fit(X_train, y_train)
        model, result_dict = evaluate_model(best_model, 'XGBoost', X_train, y_train, X_test, y_test)
        r2_base.append(result_dict['r2_val'])
        mae_base.append(result_dict['mae_val'])
        rmse_base.append(result_dict['rmse_val'])
        #result_dict

    #    onehot model

    r2_onehot = []
    mae_onehot = []
    rmse_onehot = []

    for p in per:
        X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X_b, y_b, train_size=p)

        #   Scaling and normalizing
        scaler = StandardScaler()

        X_train = normalize(scaler.fit_transform(X_train_unscaled))
        X_test = normalize(scaler.transform(X_test_unscaled))

        best_model = search.best_estimator_.fit(X_train, y_train)
        model, result_dict = evaluate_model(best_model, 'XGBoost', X_train, y_train, X_test, y_test)
        r2_onehot.append(result_dict['r2_val'])
        mae_onehot.append(result_dict['mae_val'])
        rmse_onehot.append(result_dict['rmse_val'])


    #   Writing results in a csv

    result_list = pd.DataFrame(
        {'percent_of_dataset_as_training': percentages,
         'mae': mae,
         'mae_base': mae_base,
         'mae_onehot': mae_onehot
        })
    result_list.to_csv(os.path.join(save_path_results, 'result {}.csv'.format(task)), index = False)