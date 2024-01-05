import warnings
import os
import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
# from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)

# 处理玩家角色表
roles = pd.read_csv('../data/role_id.csv')

# 在角色信息上面拼接日期信息（2号至8号）
dfs = []
for i in range(2, 9):
    tmp = roles.copy()
    tmp['day'] = i
    dfs.append(tmp)
data = pd.concat(dfs).reset_index(drop=True)

# 处理货币消耗表
consume = pd.read_csv('../data/role_consume_op.csv')
consume['dt'] = pd.to_datetime(consume['dt'])
consume['day'] = consume['dt'].dt.day

# 将货币消耗按天合并，统计每个用户每天消耗每种货币的频次和总数
for i in range(1, 5):
    for m in ['count', 'sum']:
        tmp = consume.groupby(['role_id', 'day'])[f'use_t{i}'].agg(m).to_frame(name=f'use_t{i}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 处理升级表

evolve = pd.read_csv('../data/role_evolve_op.csv')
evolve['dt'] = pd.to_datetime(evolve['dt'])
evolve['day'] = evolve['dt'].dt.day
# 新增列统计升级数
evolve['n_level_up'] = evolve['new_lv'] - evolve['old_lv']
evolve = evolve.rename(columns={'num': 'lv_consume_item_num'})

# 统计每位用户每天使用的升级道具的种类情况
for col in ['type', 'item_id']:
    for m in ['count', 'nunique']:
        tmp = evolve.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 统计每位用户每天使用升级道具量和升级情况
for col in ['lv_consume_item_num', 'n_level_up']:
    for m in ['sum', 'mean']:
        tmp = evolve.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 处理副本表

fb = pd.read_csv('../data/role_fb_op.csv')
fb['dt'] = pd.to_datetime(fb['dt'])
fb['day'] = fb['dt'].dt.day
# 新增列统计副本花费时间
fb['fb_used_time'] = fb['finish_time'] - fb['start_time']

# 统计每位用户每天挑战的副本种类情况
for col in ['fb_id', 'fb_type']:
    for m in ['count', 'nunique']:
        tmp = fb.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 统计每位用户每天挑战的副本的挑战时间和获取经验情况
for col in ['fb_used_time', 'exp']:
    for m in ['sum', 'mean']:
        tmp = fb.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 统计每位用户每天挑战副本的最后结果情况（分为0、1、2）
tmp = fb.groupby(['role_id', 'day'])['fb_result'].value_counts().reset_index(name='fb_result_count')
for i in [0, 1, 2]:
    tt = tmp[tmp['fb_result'] == i]
    tt.columns = list(tt.columns[:-1]) + ['fb_result%d_count' % i]
    data = data.merge(tt[['role_id', 'day', 'fb_result%d_count' % i]], on=['role_id', 'day'], how='left')

# 处理任务系统表

mission = pd.read_csv('../data/role_mission_op.csv')
mission['dt'] = pd.to_datetime(mission['dt'])
mission['day'] = mission['dt'].dt.day

# 统计每位用户每天完成任务的种类情况
for col in ['mission_id', 'mission_type']:
    for m in ['count', 'nunique']:
        tmp = mission.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 处理玩家离线表
offline = pd.read_csv('../data/role_offline_op.csv')
offline['dt'] = pd.to_datetime(mission['dt'])
offline['day'] = offline['dt'].dt.day
# 新增列统计在线时间
offline['online_durations'] = offline['offline'] - offline['online']

# 统计每位用户每天下线原因和下线时所在地图情况
for col in ['reason', 'map_id']:
    for m in ['count', 'nunique']:
        tmp = offline.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
# 统计每位用户每天下线时的在线时长情况
for col in ['online_durations']:
    for m in ['mean', 'sum']:
        tmp = offline.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 处理付费表
pay = pd.read_csv('../data/role_pay.csv')
pay['dt'] = pd.to_datetime(pay['dt'])
pay['day'] = pay['dt'].dt.day
# 统计每位用户每天付费金额
tmp = pay.groupby(['role_id', 'day'])['pay'].agg('sum').to_frame(name='pay_sum_day').reset_index()
data = data.merge(tmp, on=['role_id', 'day'], how='left')
data['pay_sum_day'].fillna(0., inplace=True)

# 验证集设置
# 用前n天行为预测第n+1天行为（滑动窗口）

# 训练集 day 2,3,4,5,6 -> 标签 day 7 pay_sum
df_train = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(2, 7)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    df_train = df_train.merge(tmp, on='role_id')

# 验证集 day 3,4,5,6,7 -> 标签 day 8 pay_sum
df_valid = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(3, 8)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    df_valid = df_valid.merge(tmp, on='role_id')

# 测试集 day 4,5,6,7,8
df_test = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(4, 9)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    df_test = df_test.merge(tmp, on='role_id')

# 标签构造

# 训练集 day == 7 pay_sum
# 验证集 day == 8 pay_sum

day7_pay = pay[pay.day == 7].copy().reset_index(drop=True)
tmp = day7_pay.groupby('role_id')['pay'].agg('sum').to_frame(name='pay').reset_index()
df_train = df_train.merge(tmp, on='role_id', how='left')
df_train['pay'].fillna(0., inplace=True)

day8_pay = pay[pay.day == 8].copy().reset_index(drop=True)
tmp = day8_pay.groupby('role_id')['pay'].agg('sum').to_frame(name='pay').reset_index()
df_valid = df_valid.merge(tmp, on='role_id', how='left')
df_valid['pay'].fillna(0., inplace=True)

df = pd.concat([df_train, df_valid, df_test]).reset_index(drop=True)

# 对付费金额进行对数变换（因为大多数付费金额为0）
df['pay_log'] = np.log1p(df['pay'])

df_train = df[:len(df_train)].reset_index(drop=True)
df_valid = df[len(df_train):len(df_train) + len(df_valid)].reset_index(drop=True)
df_test = df[-len(df_test):].reset_index(drop=True)

params = {
    'objective': 'regression',
    'metric': {'rmse'},
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': 16,
    'num_leaves': 32,
    'feature_fraction': 0.70,
    'subsample': 0.75,
    'seed': 36,
    'num_iterations': 3000,
    'nthread': -1,
    'verbose': -1
}

features = [col for col in df_train.columns if col not in ['role_id', 'pay', 'pay_log']]


def train(df_train, df_valid, label, params, features):
    train_label = df_train[label].values
    train_feat = df_train[features]

    valid_label = df_valid[label].values
    valid_feat = df_valid[features]
    gc.collect()

    trn_data = lgb.Dataset(train_feat, label=train_label)
    val_data = lgb.Dataset(valid_feat, label=valid_label)
    clf = lgb.train(params,
                    trn_data,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=50,
                    early_stopping_rounds=100)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    fold_importance_df = fold_importance_df.sort_values(by='importance', ascending=False)
    # 打印最重要的前30个特征
    print(fold_importance_df[:30])
    #     fold_importance_df.to_csv(f"importance_df.csv", index=None)
    df_valid['{}_preds'.format(label)] = clf.predict(valid_feat, num_iteration=clf.best_iteration)
    # 负值修正（因为充值金额最小为0）
    df_valid['{}_preds'.format(label)] = df_valid['{}_preds'.format(label)].clip(lower=0.)

    # 用np.expm1反向转换，因为之前对充值金额进行了对数变换
    result = mean_squared_log_error(np.expm1(df_valid[label]),
                                    np.expm1(df_valid['{}_preds'.format(label)]))

    #     plot(df_valid[label])
    #     plot(df_valid['{}_preds'.format(label)])
    #     show()
    # 结果可视化
    # plot(np.expm1(df_valid[label]))
    # plot(np.expm1(df_valid['{}_preds'.format(label)]))
    # show()
    plt.plot(np.expm1(df_valid[label]), label='Actual Pay', color='blue')

    # 绘制模型预测的充值金额的曲线
    plt.plot(np.expm1(df_valid['{}_preds'.format(label)]), label='Predicted Pay', color='orange')

    # 添加坐标轴标签
    plt.xlabel("Player Index")
    plt.ylabel("Pay Amount")  # 这里根据实际单位修改

    # 添加图表标题
    plt.title("Comparison of Actual and Predicted Pay Amounts")

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()

    return clf, result


clf_valid, result_valid = train(df_train, df_valid, 'pay_log', params, features)

print('rmsle score', np.sqrt(result_valid))

# 用 4,5,6,7,8 重新训练模型

params['num_iterations'] = clf_valid.best_iteration
clf_test, _ = train(df_valid, df_valid, 'pay_log', params, features)
print('rmsle score', np.sqrt(_))
# 因为进行了对数变换，这里用np.expm1将其转换回来
df_test['pay'] = np.expm1(clf_test.predict(df_test[features]))
# 负值修正
df_test['pay'] = df_test['pay'].clip(lower=0.)
print(df_test['pay'].describe())

sub = pd.read_csv('../data/submission_sample.csv')
sub_df = df_test[['role_id', 'pay']].copy()
sub = sub[['role_id']].merge(sub_df, on='role_id', how='left')
sub[['role_id', 'pay']].to_csv('submission.csv', index=False)
