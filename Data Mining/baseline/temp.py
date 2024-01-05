import pandas as pd
import numpy as np
import sys
# LightGBM是实现GBDT算法(梯度提升决策树，一种回归算法)的框架
# 其中LGBMRegressor类是专门用于解决回归问题的模型(连续值的回归)
from lightgbm import LGBMRegressor as LGBR
from sklearn.model_selection import GridSearchCV
import warnings
# 后面用KFold测测性能
from sklearn.model_selection import KFold
from preprocess import preprocess_data

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore")
clf = LGBR(verbose=-1)
cv_params = {
    "max_depth": [3, 7, 13],
    "n_estimators": [300, 700, 1000],
    "learning_rate": [0.005, 0.01, 0.02]
}
role_id,x_test, x_tr, x_te, y_tr, y_te = preprocess_data()
eval_set = [(x_te, y_te)]
# GridSearchCV选取最优参数(可优化?)
# 选取依据:均方误差
clf_search = GridSearchCV(clf, param_grid=cv_params, scoring='neg_mean_squared_error',
                          n_jobs=-1, cv=5)

clf_search.fit(x_tr, y_tr, eval_set=eval_set)
print(clf_search.best_params_)
