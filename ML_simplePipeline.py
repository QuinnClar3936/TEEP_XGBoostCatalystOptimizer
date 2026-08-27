import config
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from numpy import nan



class PitchaiPreprocess(TransformerMixin, BaseEstimator):
    def __init__(self, inter_min, inter_max, training_vars):
        self.inter_min = inter_min #currently dont contribute- written in config -> fix when relocating classes to config
        self.inter_max = inter_max
        self.training_vars = training_vars
        pass

    def fit(self, X, y=None):
        return(self)
    
    def transform(self, X):        
        #build interactions first from raw elemental ratios
        X_inters = config.create_interactions(X, self.training_vars)
        return(X_inters)

class PitchaiSelect(TransformerMixin, BaseEstimator):
    def __init__(self, k_features):
        self.k = k_features
        pass

    def fit(self, X, y=None):
        self.selector_ = SelectKBest(f_regression, k=self.k)
        self.selector_.fit(X, y)
        mask = self.selector_.get_support()
        names_unsorted = X.columns[mask].tolist()
        self.selected_feature_names_ = sorted(names_unsorted)
        return self
    
    def transform(self, X):
        X_best = X[self.selected_feature_names_]
        return X_best



#init
estimators = 1300
tree_depth = 1
parallel_trees = 1
feature_subsample = 0.3
subsample_ratio = 0.2
child_weight = 5
alpha_val = 0.14 #L1/Lasso
lambda_val = 0.80 #L2/Ridge
learningrate = 0.2
max_idempot = 10
k_features = 25
min_inter = 2 #not 1, as that is included in the features
max_inter = 5
training_vars = ['Fe', 'Co', 'Cr', 'Mn', 'Cu']

bst = xgb.XGBRegressor(
    n_estimators=estimators, 
    max_depth=tree_depth, 
    subsample=subsample_ratio,
    colsample_bytree=feature_subsample,
    num_parallel_tree=parallel_trees,
    min_child_weight=child_weight,
    reg_alpha=alpha_val,
    reg_lambda=lambda_val,
    learning_rate=learningrate, 
    objective='reg:squarederror', 
    #early_stopping_rounds=max_idempot,
    random_state=42
    )

yeo_johnson = PowerTransformer(method='yeo-johnson').set_output(transform='pandas')

#preprocess
df_train_unfixed = pd.read_csv('ML_paper_data_HE.csv')

    #generate data split
df_train_unfixed, df_test_unfixed = train_test_split(df_train_unfixed, test_size=0.20, random_state=42)
print('train data set has got {} rows and {} columns'.format(df_train_unfixed.shape[0],df_train_unfixed.shape[1]))
print('test data set has got {} rows and {} columns'.format(df_test_unfixed.shape[0],df_test_unfixed.shape[1]))

    #collection of target variable
y_train = df_train_unfixed["Overpotential"]
y_test = df_test_unfixed["Overpotential"]

df_train_unfixed = df_train_unfixed.drop("Overpotential", axis=1)
df_test_unfixed = df_test_unfixed.drop("Overpotential", axis=1)

    #train data pipeline
PitchaiPipeline = Pipeline([
    ('split', PitchaiPreprocess(min_inter, max_inter, training_vars)),
    ('yeo-johnson', yeo_johnson),
    ('select_k_best', PitchaiSelect(k_features)),
    ('regressor', bst)
])

#train
PitchaiPipeline.fit(df_train_unfixed, y_train)

    #predict
train_preds = PitchaiPipeline.predict(df_train_unfixed)
test_preds  = PitchaiPipeline.predict(df_test_unfixed)

    #score
train_r2 = PitchaiPipeline.score(df_train_unfixed, y_train)
test_r2  = PitchaiPipeline.score(df_test_unfixed, y_test)
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

#validation
    #recover feature names and selector
fitted_select_step = PitchaiPipeline.named_steps['select_k_best']
selected_feature_names = fitted_select_step.selected_feature_names_
selector = fitted_select_step.selector_

    #find best composition
best_comp, result = config.locate_optimal_comp(
    bst=bst,
    transformer=yeo_johnson,
    selector=selector,
    selected_feature_names=selected_feature_names,
    df_train_unfixed=df_train_unfixed
    )

print("Predicted minimum overpotential:", result)
print("Optimal composition:", best_comp)
print(selected_feature_names)

known_opt = pd.DataFrame([{
    'Fe': 0.15, 'Co': 0.10, 'Cr': 0.30, 'Mn': 0.30, 'Cu': 0.15
}])

print(f"Known Opt: {PitchaiPipeline.predict(known_opt)}")