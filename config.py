#config and function definition
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

from random import  uniform, randrange

from numpy import nan

from itertools import combinations

from scipy.optimize import differential_evolution, LinearConstraint

import pandas as pd
import numpy as np



#default params
estimators = 1000
tree_depth = 2
parallel_trees = 1
feature_subsample = 0.25
subsample_ratio = 0.5
child_weight = 3
alpha_val = 0.55 #L1/Lasso
lambda_val = 2.0 #L2/Ridge
learningrate = 0.3
max_idempot = 10
k_features = 25
min_inter = 2 #not 1, as that is included in the features
max_inter = 5
tolerance = 0.01



#dicts
opt_scoring = {
    "rmse":make_scorer(root_mean_squared_error, greater_is_better=False),
    "r2":make_scorer(r2_score)
}

init_param_grid = {
    "model__n_estimators": [1100, 1300, 1500],
    "model__max_depth": [1],
    "model__subsample": [0.1, 0.2, 0.3],
    "model__min_child_weight": [4.0, 5.0, 6.0],
    "model__colsample_bytree": [0.2, 0.3, 0.4],
    "model__learning_rate": [0.1, 0.2, 0.3],
    "model__alpha": [0.1, 1.0, 2.0],
    "model__lambda": [0.1, 1.0, 2.0],
}

param_grid_bounds = {
    "model__n_estimators": [100, 2000, int],
    "model__max_depth": [1, 7, int],
    "model__subsample": [0.05, 0.95, float],
    "model__min_child_weight": [0.0, 10.0, float],
    "model__colsample_bytree": [0.05, 0.95, float],
    "model__learning_rate": [0.05, 0.95, float],
    "model__alpha": [0.1, 5.0, float],
    "model__lambda": [0.1, 5.0, float],
}

opt_estimator = Pipeline([
    ("select", SelectKBest(f_regression, k=k_features)),
    ("model", xgb.XGBRegressor(tree_method="hist", random_state=42, n_jobs=1))
])



def predict_overpotential_penalized(comp, df_train_unfixed, y_train, transformer, selector, selected_feature_names, bst, k=3, disagreement_weight=1.0):
    #collect actual experimentally recorded values
    train_vecs = df_train_unfixed[['Fe', 'Co', 'Cr', 'Mn', 'Cu']].values
    comp_df = pd.DataFrame([comp], columns=['Fe', 'Co', 'Cr', 'Mn', 'Cu'])
    comp_df_t = transformer.transform(comp_df)          #include yeo-johnson transform
    comp_inters = create_interactions(comp_df_t, comp_df_t)   # use transformed df
    comp_poly_selected = selector.transform(comp_inters)
    comp_poly_df = pd.DataFrame(comp_poly_selected, columns=selected_feature_names)
    #make prediction
    pred = bst.predict(comp_poly_df)[0]
    #penalize for distance from experimental overpotential values
    dists = np.linalg.norm(train_vecs - np.array(comp), axis=1)
    knn_idx = np.argsort(dists)[:k]
    knn_actual_mean = y_train.iloc[knn_idx].mean()
    disagreement = abs(pred - knn_actual_mean)
    return pred + disagreement_weight * disagreement

def create_polynomial_inters(x_input, passed_training_vars, min_inter_order=min_inter, max_inter_order=max_inter):
    poly_inters = pd.DataFrame(index=x_input.index)
    for feature in sorted(passed_training_vars):
        for order_p in range(min_inter_order, max_inter_order + 1):
            name = f"{feature}{order_p}"
            poly_inters[name] = x_input[feature] ** order_p
    return poly_inters

def create_self_inters(x_input, passed_training_vars, min_inter_order=min_inter, max_inter_order=max_inter):
    self_inters = pd.DataFrame(index=x_input.index)
    for order_s in range(min_inter_order, max_inter_order + 1):
        for combo in combinations(sorted(passed_training_vars), order_s):
            name = '*'.join(combo)
            product = x_input[combo[0]].copy()
            for f in combo[1:]:
                product *= x_input[f]
            self_inters[name] = product
    return self_inters

def collect_first_inters(x_input, passed_training_vars, inter_order=1): 
    first_inters = pd.DataFrame(index=x_input.index)
    for feature in sorted(passed_training_vars):
        for order_f in range(1, inter_order+1):
            name = f"{feature}{order_f}"
            first_inters[name] = x_input[feature] ** order_f
    return first_inters

def create_interactions(x_set, passed_training_variables):
    first_o_inters = collect_first_inters(x_set, passed_training_variables)
    poly_inters = create_polynomial_inters(x_set, passed_training_variables)
    self_inters = create_self_inters(x_set, passed_training_variables)
    higher_inters = pd.concat([poly_inters, self_inters], axis=1)
    complete_inter_set = pd.concat([first_o_inters, higher_inters], axis=1)
    return complete_inter_set

def single_opt_cycle(param_grid, x_train, y_train):
    gcv = RandomizedSearchCV(
        estimator=opt_estimator, 
        param_distributions=param_grid,
        scoring=opt_scoring, 
        n_jobs=12, 
        refit="r2", 
        cv=5, 
        verbose=0, 
        pre_dispatch='2*n_jobs', 
        error_score=nan, 
        return_train_score=False
    )
    gcv.fit(x_train, y_train)
    gcv.cv_results_
    print("Best Parameters:", gcv.best_params_)
    print("Best CV Score:", gcv.best_score_)
    return(gcv.best_params_, gcv.best_score_)

def expand_from_opt(pseudo_opt, prev_param):
    expanded_params = dict(pseudo_opt)
    for key, best_value in pseudo_opt.items():
        prev_values = prev_param[key]
        lo_bound, hi_bound, cast = param_grid_bounds[key]
        param_radius = 0.40 * (max(prev_values) - min(prev_values))
        if cast is int:
            param_radius = max(param_radius, 1)
        upper_value = best_value + param_radius
        lower_value = best_value - param_radius
        upper_value = min(upper_value, hi_bound)
        lower_value = max(lower_value, lo_bound)
        #avoid duplicate values at a bound
        expanded_params[key] = sorted({cast(lower_value), cast(best_value), cast(upper_value)})
    return(expanded_params)

def create_regression(tuned_params, x_train, y_train, x_test, y_test, iteration):
    #Create regression model with optimized hyperparams
    params_dict = tuned_params  
    print(f"Tuned Parameters: {params_dict}")

    bst = xgb.XGBRegressor(
        n_estimators=params_dict['model__n_estimators'], 
        max_depth=params_dict['model__max_depth'], 
        subsample=params_dict['model__subsample'],
        colsample_bytree=params_dict['model__colsample_bytree'],
        num_parallel_tree=parallel_trees,
        min_child_weight=params_dict['model__min_child_weight'],
        reg_alpha=params_dict['model__alpha'],
        reg_lambda=params_dict['model__lambda'],
        learning_rate=params_dict['model__learning_rate'], 
        objective='reg:squarederror', 
        #early_stopping_rounds=max_idempot,
        random_state=42
        )

    bst.fit(x_train, y_train)

    test_preds = bst.predict(x_test)
    train_preds = bst.predict(x_train)

    train_rmse = root_mean_squared_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)
    print(f'Regression Iter {iteration+1} Output Evaluation:')
    print(f"Train RMSE: {train_rmse:.4e}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4e}")
    print(f"Test R²: {test_r2:.4f}")

    return(bst, test_r2, train_r2)

def generate_hparam_interval(param_bounds=param_grid_bounds):
    hparam_interval = {}
    for key in param_bounds:
        low, high, cast = param_bounds[key]
        if cast == int:
            first = randrange(low, high)
            second = randrange(low, high)
            middle = int((first+second)/2)
        if cast == float:
            first = uniform(low, high)
            second = uniform(low, high)
            middle = (first+second)/2
        upper = max(first, second)
        lower = min(first, second)
        hparam_interval[key] = [lower, middle, upper]
    return(hparam_interval)

def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in compositions(total - i, parts - 1):
            yield (i,) + rest

def locate_optimal_comp(bst, transformer, selector,  selected_feature_names, df_train_unfixed):
    n_steps = 500
    elements = ['Fe', 'Co', 'Cr', 'Mn', 'Cu']

    grid_int = list(compositions(n_steps, len(elements)))

    grid_df = pd.DataFrame(np.array(grid_int) / n_steps, columns=elements)

    # restrict to the model's interpolation domain
    bounds = {el: (df_train_unfixed[el].min(), df_train_unfixed[el].max()) for el in elements}
    mask = np.ones(len(grid_df), dtype=bool)
    for el in elements:
        lo, hi = bounds[el]
        if lo < 0.10:
            lo = 0.10
        mask &= (grid_df[el] >= lo) & (grid_df[el] <= hi)
    grid_df = grid_df[mask].reset_index(drop=True)

    grid_raw_inters = create_interactions(grid_df, elements)
    grid_transformed = transformer.transform(grid_raw_inters)
    grid_selected = selector.transform(grid_transformed)
    grid_selected_df = pd.DataFrame(grid_selected, columns=selected_feature_names)

    preds = bst.predict(grid_selected_df)

    best_idx  = np.argmin(preds)
    best_comp = grid_df.iloc[best_idx]
    best_pred = preds[best_idx]

    return(best_comp, best_pred)