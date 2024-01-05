import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings


def preprocess_data():
    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    warnings.filterwarnings("ignore")
    # 表1(角色信息表)、表6(角色上下线表)暂不考虑，后续应加上
    # 各表数据均存在过于离谱的值
    df1 = pd.read_csv("../data/role_id.csv")  # 角色信息
    df2 = pd.read_csv("../data/role_consume_op.csv")  # 货币使用
    df3 = pd.read_csv("../data/role_evolve_op.csv")  # 角色升级
    df4 = pd.read_csv("../data/role_fb_op.csv")  # 挑战副本
    df5 = pd.read_csv("../data/role_mission_op.csv")  # 完成任务
    # df6 = pd.read_csv("../data/role_offline_op.csv") #角色上下线
    df7 = pd.read_csv("../data/role_pay.csv")  # 充值记录
    df8 = pd.read_csv("../data/submission_sample.csv")  # 提交样例

    # 白嫖不充钱的统统pass(后续需要处理，因为一共4000人，仅1200人充值)
    role_id = list(df7['role_id'].unique())
    all_id = list(df1['role_id'].unique())
    df2 = df2[df2["role_id"].isin(role_id)]
    df3 = df3[df3["role_id"].isin(role_id)]
    df4 = df4[df4["role_id"].isin(role_id)]
    df5 = df5[df5["role_id"].isin(role_id)]

    # 仅考虑天(mtime均忽略)
    df2["day"] = df2["dt"].apply(lambda x: x[9:10])
    df3["day"] = df3["dt"].apply(lambda x: x[9:10])
    df4["day"] = df4["dt"].apply(lambda x: x[9:10])
    df5["day"] = df5["dt"].apply(lambda x: x[9:10])
    df7["day"] = df7["dt"].apply(lambda x: x[9:10])

    # 充值表预处理
    df7 = df7.iloc[:, [0, 1, 4]]

    # ds字典:每个用户一周中每天的充值量
    ds = dict()
    for i in role_id:
        ds[i] = [0 for j in range(7)]
    for i in df7.values:
        ds[i[0]][int(i[2]) - 2] += i[1]
    print("充值表预处理完成")
    # 货币消耗表预处理
    yhl_2 = pd.DataFrame()
    for i in role_id:
        df2_role = df2[df2["role_id"] == i]
        for j in ["2", "3", "4", "5", "6", "7", "8"]:
            cols = df2_role.columns
            df2_role_days = df2_role[df2_role["day"] == j]
            if not list(df2_role_days.values):
                df2_role_days = pd.DataFrame([[i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, j]])
                df2_role_days.columns = cols
            level = df2_role_days["level"].values[-1]  # 获取每天的最终等级
            # t3、t4全是0，废数据
            use_t1 = df2_role_days["use_t1"].sum()
            use_t2 = df2_role_days["use_t2"].sum()
            remain_t1 = df2_role_days["remain_t1"].values[-1]
            remain_t2 = df2_role_days["remain_t2"].values[-1]

            df2_yhl = pd.DataFrame([[i, level, use_t1, use_t2,
                                     remain_t1, remain_t2, j]])
            yhl_2 = pd.concat([yhl_2, df2_yhl], axis=0)
    # 等级与使用货币的关系
    df_yhl_2 = yhl_2.iloc[:, [1, 2, 3, 4, 5]].values
    df_yhl_2 = df_yhl_2.reshape(-1, 35)
    print("货币消耗表预处理完成")
    # 升级表预处理
    yhl_3 = pd.DataFrame()
    for i in role_id:
        df3_role = df3[df3["role_id"] == i]
        for j in ["2", "3", "4", "5", "6", "7", "8"]:
            cols = df3_role.columns
            df3_role_days = df3_role[df3_role["day"] == j]
            if not list(df3_role_days.values):
                df3_role_days = pd.DataFrame([[i, 0, 0, 0, 0, 0, 0, 0, j]])
                df3_role_days.columns = cols
            fea = []
            # 记录该用户该天消耗的道具类型
            type_ = set(list(df3_role_days["type"].values))
            fea_type = [0 for k in range(11)]
            for k in type_:
                fea_type[k - 1] = 1
            num = [df3_role_days["num"].sum()]  # 直接把所有类的道具数量全部加在一起算了
            fea += [i]
            fea += fea_type
            fea += num
            fea += [j]
            df3_yhl = pd.DataFrame([fea])
            yhl_3 = pd.concat([yhl_3, df3_yhl], axis=0)
    # 记录道具类型组合和总消耗量
    df_yhl_3 = yhl_3.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values
    df_yhl_3 = df_yhl_3.reshape(-1, 84)
    print("升级表预处理完成")
    # 副本表预处理
    yhl_4 = pd.DataFrame()
    for i in role_id:
        df4_role = df4[df4["role_id"] == i]
        for j in ["2", "3", "4", "5", "6", "7", "8"]:
            cols = df4_role.columns
            df4_role_days = df4_role[df4_role["day"] == j]
            if not list(df4_role_days.values):
                df4_role_days = pd.DataFrame([[i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, j]])
                df4_role_days.columns = cols

            fea = []
            # 仅考虑副本类型
            role_level = df4_role_days["role_level"].values[-1]
            type_ = set(list(df4_role_days["fb_type"].values))
            fea_type = [0 for k in range(2)]
            for k in type_:
                fea_type[k] = 1
            day_times = df4_role_days["day_times"].sum()
            challenge_times = df4_role_days["challenge_times"].sum()
            exp = df4_role_days["exp"].sum()

            fea += [i]
            fea += [role_level]
            fea += fea_type
            fea += [day_times]
            fea += [challenge_times]
            fea += [exp]
            fea += [j]
            df4_yhl = pd.DataFrame([fea])
            yhl_4 = pd.concat([yhl_4, df4_yhl], axis=0)

    df_yhl_4 = yhl_4.iloc[:, [1, 2, 3, 4, 5, 6]].values
    df_yhl_4 = df_yhl_4.reshape(-1, 42)
    print("副本表预处理完成")
    # 任务表预处理
    yhl_5 = pd.DataFrame()
    for i in role_id:
        df5_role = df5[df5["role_id"] == i]
        for j in ["2", "3", "4", "5", "6", "7", "8"]:
            cols = df5_role.columns
            df5_role_days = df5_role[df5_role["day"] == j]
            if not list(df5_role_days.values):
                df5_role_days = pd.DataFrame([[i, 0, 0, 0, 0, 0, 0, 0, j]])
                df5_role_days.columns = cols

            # 此处处理待定(大多用户每天普遍就完成一个任务)
            role_level = df5_role_days["role_level"].values[-1]
            mission_type = df5_role_days["mission_type"].values[-1]
            mission_status = df5_role_days["mission_status"].values[-1]
            total_times = df5_role_days["total_times"].values[-1]

            df5_yhl = pd.DataFrame([[i, role_level, mission_type, mission_status,
                                     total_times, j]])
            yhl_5 = pd.concat([yhl_5, df5_yhl], axis=0)

    df_yhl_5 = yhl_5.iloc[:, [1, 2, 3, 4]].values
    df_yhl_5 = df_yhl_5.reshape(-1, 28)
    print("任务表预处理完成")
    # 滑动窗口
    x, y = [], []
    for idx, i in enumerate(role_id):
        # 前6天的特征数据
        x.append(ds[i][:-1] + list(df_yhl_2[idx])[:30] + list(df_yhl_3[idx])[:72] +
                 list(df_yhl_4[idx])[:36] + list(df_yhl_5[idx])[:24])
        # 第7天的充值量(即目标数据)
        y.append(ds[i][-1])
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)

    scaler = StandardScaler()
    scaler = scaler.fit(x)
    x = scaler.transform(x)

    x_test = []
    for idx, i in enumerate(role_id):
        # 用2到7天预测第8天
        x_test.append(ds[i][1:] + list(df_yhl_2[idx])[5:] + list(df_yhl_3[idx])[12:] +
                      list(df_yhl_4[idx])[6:] + list(df_yhl_5[idx])[4:])
    print(np.array(x_test).shape)
    x_test = scaler.transform(x_test)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=126)

    return role_id, x_test, x_tr, x_te, y_tr, y_te, all_id

# if __name__ == '__main__':
#     df2, df3, df4, df5, df7, x, y, x_test, x_tr, x_te, y_tr, y_te = preprocess_data()
