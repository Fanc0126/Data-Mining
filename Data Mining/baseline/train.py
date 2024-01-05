from lightgbm import LGBMRegressor
import joblib
from preprocess import *

role_id, x_test, x_tr, x_te, y_tr, y_te, all_id = preprocess_data()

# best_params = {'learning_rate': 0.005, 'max_depth': 13, 'n_estimators': 300}
# model = LGBMRegressor(**best_params, verbose=-1)
# model.fit(x_tr, y_tr, eval_set=[(x_te, y_te)], verbose=True)
# model_filename = 'trained_model.pkl'
# joblib.dump(model, model_filename)

model_filename = 'trained_model.pkl'
model2 = joblib.load(model_filename)
predictions = model2.predict(x_test)


# results = pd.DataFrame({'role_id': role_id, 'pay': predictions})
# results.to_csv('predictions.csv', index=False)
# 还要处理一下小数，甚至有负数
def discretize_value(value):
    values = [6, 30, 68, 128, 198, 298, 648]
    return min(values, key=lambda x: abs(x - value))


# 将预测结果离散化
discretized_predictions = [discretize_value(value) for value in predictions]

# 创建包含role_id和预测结果的DataFrame
result_df = pd.DataFrame({'role_id': role_id, 'pay': discretized_predictions})

for id in all_id:
    if id not in result_df['role_id'].values:
        result_df = result_df.append({'role_id': id, 'pay': 0}, ignore_index=True)

# 修改数据类型
result_df['role_id'] = result_df['role_id'].astype(str)
result_df['pay'] = result_df['pay'].astype(float)

# 保存为CSV文件
result_df.to_csv('sub.csv', index=False)
