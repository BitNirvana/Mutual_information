import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)  # 降序排列，默认升序
    return mi_scores


plt.style.use("seaborn-whitegrid")

df = pd.read_csv("t_user_score_10W.csv", encoding='gbk')
# print(df.head())

X = df.copy()
X = X.drop(['t_user_score.userid', 't_user_score.city_id', 't_user_score.city_name', 't_user_score.region_id',
            't_user_score.region_name', 't_user_score.cust_id', 't_user_score.ds'], axis=1)
y = X.pop("t_user_score.score")

# print(X.head())
# print(y.head())

# 分类的标签编码
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# 所有离散特性现在都应该具有整数数据类型（在使用MI之前请仔细检查此项！）
discrete_features = X.dtypes == int


mi_scores = make_mi_scores(X, y, discrete_features)
print(mi_scores)
# print(mi_scores[::3])  # show a few features with their MI scores
