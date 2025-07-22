from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, median_absolute_error, r2_score,mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import multiprocessing

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

threads = multiprocessing.cpu_count()-2

df = pd.read_csv('housing_price.csv')
keep_cols = ['district',]

X = df[keep_cols]

y = df['price']

print(X.shape, y.shape)
print(X.columns)
print(y.head())

def clean_data(X):
    X.fillna(-1, inplace=True)
    return X

X_clean = clean_data(X.copy())
# print(X_clean.head())
X_clean.head()

param_grids={
    'Ridge':{
        'ridge__alpha':[ 0.01, 0.1, 1, 10, 100,]
    },
    'Lasso':{
        'lasso__alpha':[ 0.01, 0.1, 1, 10, 100]
    },
    'ElasticNet':{
        'elasticnet__alpha':[ 0.01, 0.1, 1, 10],
        'elasticnet__l1_ratio':[0.1,0.3,0.5,0.7,0.9]
    },
    'SVR':{
        'svr__C': [0.01,0.1, 1, 10, 100],
        'svr__kernel': ['rbf'],
        'svr__gamma': [0.01,0.1, 1, 10, 100]
    },
    'Decision_Tree':{
        'decision_tree__max_depth': [2, 4, 6, 8, 10, 12, 14],
        'decision_tree__min_samples_split': [2, 5, 10, 20]
    },
    'Random_Forest':{
        'random_forest__max_depth': [2, 4, 6, 8, 10, 12, 14],
        'random_forest__min_samples_split': [2, 5, 10, 20],
        'random_forest__n_estimators': [10, 50, 100, 200]
    },
    'xgb':{
        'xgb__max_depth': [2, 4, 6, 8, 10, 12, 14],
        'xgb__min_child_weight': [1, 2, 3, 4, 5],
        'xgb__learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
        'xgb__n_estimators': [10, 50, 100, 200]
    }
}
random_state= 2025
best_params = {
    'Ridge': {'ridge__alpha': [100]},
    'Lasso': {'lasso__alpha': [100]},
    'ElasticNet': {'elasticnet__alpha': [0.1], 'elasticnet__l1_ratio': [0.5]},
    'Decision_Tree': {'decision_tree__max_depth': [6], 'decision_tree__min_samples_split': [20]},
    'Random_Forest': {'random_forest__max_depth': [14], 'random_forest__min_samples_split': [2], 'random_forest__n_estimators': [200]},
    'xgb': {'xgb__learning_rate': [0.1], 'xgb__max_depth': [4], 'xgb__min_child_weight':[2], 'xgb__n_estimators': [200]}
}

param_grids = best_params

# Best Params
# N/A
# {'ridge__alpha': 100}
# {'lasso__alpha': 100}
# {'elasticnet__alpha': 0.1, 'elasticnet__l1_ratio': 0.5}
# {'decision_tree__max_depth': 6, 'decision_tree__min_samples_split': 20}
# {'random_forest__max_depth': 14, 'random_forest__min_samples_split': 2, 'random_forest__n_estimators': 200}
# {'xgb__learning_rate': 0.1, 'xgb__max_depth': 4, 'xgb__min_child_weight': 2, 'xgb__n_estimators': 200}

models={
    'OLS': LinearRegression(),
    'Ridge': Ridge(random_state=random_state),
    'Lasso': Lasso(random_state=random_state),
    'ElasticNet': ElasticNet(random_state=random_state),
    # 'SVR':SVR(),
    'Decision_Tree': DecisionTreeRegressor(random_state=random_state),
    'Random_Forest': RandomForestRegressor(random_state=random_state),
    'xgb': xgb.XGBRegressor(random_state=random_state)
    
}
results = pd.DataFrame(columns=['Model', 'Best Params', 'RMSE', 'MAE', 'MeAE', 'R-squared'])
coef_df = pd.DataFrame()

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
# 其他代码保持不变
print('Training set size:', X_train.shape)
# 在循环中，修改模型名称与步骤名称的一致性
for name, model in models.items():
    print(f'Training {name} model...')
    if name in param_grids:
        # 创建管道
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (name.lower(), model)
        ])
        # 设置网格搜索
        grid = GridSearchCV(pipeline, param_grid=param_grids[name], cv=5, n_jobs=-1, scoring='neg_root_mean_squared_error', refit=True)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        # 创建管道
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (name.lower(), model)
        ])
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = 'N/A'
    print(f'Best {name} model parameters: {best_params}')
    
    # 在验证集上进行预测
    y_pred = best_model.predict(X_test)

    # 计算性能指标
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

    # 将结果添加到 DataFrame
    new_row = pd.DataFrame({
        'Model': [name],
        'Best Params': [best_params],
        'RMSE': [rmse],
        'MAE': [mae],
        'MeAE': [medae],
        'R-squared': [r2],
        'Adj. R-squared': [adj_r2],
        'MAPE': [mape]
    })
    results = pd.concat([results, new_row], ignore_index=True)

    # 如果模型具有系数属性，保存系数
    if hasattr(best_model.named_steps[name.lower()], 'coef_'):
        coef = best_model.named_steps[name.lower()].coef_
        coef_df[name] = coef

# 设置系数 DataFrame 的索引
coef_df.index = X_clean.columns
coef_df.index.name = 'Variables'

# 输出结果
print(results)
print(coef_df)

import os
coef_df_tr = coef_df.transpose()
os.makedirs('./output', exist_ok=True)
coef_df_tr.to_csv(f'./output/coef_df_tr.csv')
results.to_csv(f'./output/results.csv')
print(f'Coef DataFrame and Results saved to output/*.csv')

