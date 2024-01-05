import pandas as pd

# 读取submission.csv文件
df = pd.read_csv('submission.csv')

# 检查role_id列的类型，如果不是str，则转换为str
if df['role_id'].dtype != 'str':
    df['role_id'] = df['role_id'].astype(str)

# 检查pay列的类型，如果不是float，则转换为float
if df['pay'].dtype != 'float':
    df['pay'] = df['pay'].astype(float)

# 保存修改后的DataFrame回到submission.csv文件
df.to_csv('submission.csv', index=False)
