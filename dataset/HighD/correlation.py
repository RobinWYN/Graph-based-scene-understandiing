import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

import seaborn as sns
from data_management.preprocess import *

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA

FIGURE_PATH = "D:/Workspace/UncertaintySafeField/paper/figures/chap02/"

PRECEDING_T = "precedingT"
FOLLOWING_T = "followingT"
TARGET_PRECEDING_T = "targetPrecedingT"
TARGET_FOLLOWING_T = "targetFollowingT"
MEAN_X_VELOCITY = "meanXVelocity"

IRRELEVENT_KEY = {
    ID,
    TARGET_SET_ID,
    TARGET_PRECEDING_ID,
    TARGET_FOLLOWING_ID,
    PRECEDING_ID,
    FOLLOWING_ID,
    LEFT_PRECEDING_ID,
    LEFT_ALONGSIDE_ID,
    LEFT_FOLLOWING_ID,
    RIGHT_PRECEDING_ID,
    RIGHT_ALONGSIDE_ID,
    RIGHT_FOLLOWING_ID,
}
LINEAR_FEATURE_LIST = [
    X,
    Y,
    X_VELOCITY,
    Y_VELOCITY,
    PRECEDING_X,
    PRECEDING_Y,
    PRECEDING_X_VELOCITY,
    FOLLOWING_X,
    FOLLOWING_Y,
    FOLLOWING_X_VELOCITY,
    TARGET_PRECEDING_X,
    TARGET_PRECEDING_Y,
    TARGET_PRECEDING_X_VELOCITY,
    TARGET_FOLLOWING_X,
    TARGET_FOLLOWING_Y,
    TARGET_FOLLOWING_X_VELOCITY,
]
NONLINEAR_FEATURE_LIST = [
    PRECEDING_T,
    FOLLOWING_T,
    TARGET_PRECEDING_T,
    TARGET_FOLLOWING_T,
    MEAN_X_VELOCITY,
]
FEATURE_LIST = LINEAR_FEATURE_LIST + NONLINEAR_FEATURE_LIST

if __name__ == "__main__":
    # df = pd.read_csv(PROCESSED_DATA_PATH + "set01_veh0030.csv")
    # plt.plot(df[X], df[Y])
    # plt.show()

    raw_df = pd.read_csv(PROCESSED_DATA_PATH + "meta.csv")  # 读入数据
    # for key in IRRELEVENT_KEY:
    #     feature_df = df.drop(key, axis=1, inplace=True)

    # 非线性特征初始化
    nonlin_feat_df = pd.DataFrame(
        np.zeros((raw_df[X].shape[0], len(NONLINEAR_FEATURE_LIST))),
        columns=NONLINEAR_FEATURE_LIST,
    )

    # 计算时距特征
    pre_true = raw_df[PRECEDING_X_VELOCITY] != 0
    nonlin_feat_df[PRECEDING_T][pre_true] = raw_df[PRECEDING_X][pre_true] / (
        raw_df[X_VELOCITY][pre_true] - raw_df[PRECEDING_X_VELOCITY][pre_true]
    )
    fol_true = raw_df[FOLLOWING_X_VELOCITY] != 0
    nonlin_feat_df[FOLLOWING_T][fol_true] = raw_df[FOLLOWING_X][fol_true] / (
        raw_df[X_VELOCITY][fol_true] - raw_df[FOLLOWING_X_VELOCITY][fol_true]
    )
    tar_pre_true = raw_df[TARGET_PRECEDING_X_VELOCITY] != 0
    nonlin_feat_df[TARGET_PRECEDING_T][tar_pre_true] = raw_df[TARGET_PRECEDING_X][
        tar_pre_true
    ] / (
        raw_df[X_VELOCITY][tar_pre_true]
        - raw_df[TARGET_PRECEDING_X_VELOCITY][tar_pre_true]
    )
    tar_fol_true = raw_df[TARGET_FOLLOWING_X_VELOCITY] != 0
    nonlin_feat_df[TARGET_FOLLOWING_T][tar_fol_true] = raw_df[TARGET_FOLLOWING_X][
        tar_fol_true
    ] / (
        raw_df[X_VELOCITY][tar_fol_true]
        - raw_df[TARGET_FOLLOWING_X_VELOCITY][tar_fol_true]
    )
    # 归一化，避免自车与周车速度相同时，时距为无穷大
    nonlin_feat_df[nonlin_feat_df > 1000] = 1000
    nonlin_feat_df[nonlin_feat_df < -1000] = -1000

    # 计算平均速度特征
    v_df = pd.DataFrame(
        raw_df[
            [
                X_VELOCITY,
                PRECEDING_X_VELOCITY,
                FOLLOWING_X_VELOCITY,
                LEFT_PRECEDING_X_VELOCITY,
                LEFT_ALONGSIDE_X_VELOCITY,
                LEFT_FOLLOWING_X_VELOCITY,
                RIGHT_PRECEDING_X_VELOCITY,
                RIGHT_ALONGSIDE_X_VELOCITY,
                RIGHT_FOLLOWING_X_VELOCITY,
            ]
        ]
    )
    # 将0替换为NaN，则计算平均值时不会考虑
    v_df = v_df.replace(0, np.nan)
    nonlin_feat_df[MEAN_X_VELOCITY] = v_df.mean(1)

    feature_df = pd.concat([raw_df[LINEAR_FEATURE_LIST], nonlin_feat_df], axis=1)
    feature_df.to_csv(PROCESSED_DATA_PATH + "/correlation.csv", index=False)
    feature_array = np.array(feature_df)

    # Bartlett's球状检验

    chi_square_value, p_value = calculate_bartlett_sphericity(feature_array)
    print(chi_square_value, p_value)

    # KMO检验
    # 检查变量间的相关性和偏相关性，取值在0-1之间；KOM统计量越接近1，变量间的相关性越强，偏相关性越弱，因子分析的效果越好。
    # 通常取值从0.6开始进行因子分析

    kmo_all, kmo_model = calculate_kmo(feature_array)
    print(kmo_all)

    # 进行标准化
    feature_array = preprocessing.scale(feature_array)

    # 求解系数相关矩阵，
    covX = np.around(np.corrcoef(feature_array.T), decimals=3)
    print(f"Cov Matrix:\n{covX}")

    # 求解特征值和特征向量，特征值已自动进行排序，由大到小
    featValue, featVec = np.linalg.eig(covX.T)  # 求解系数相关矩阵的特征值和特征向量
    print(f"\nFeature Value:\n{featValue}\nFeature Vector:\n{featVec}")

    # # 对特征值进行排序并输出 降序
    # featValue = sorted(featValue)[::-1]
    # # print(featValue)

    # # 绘制特征值大小散点图和折线图
    # plt.scatter(range(1, feature_array.shape[1] + 1), featValue)
    # plt.plot(range(1, feature_array.shape[1] + 1), featValue)
    # # plt.title("各特征向量对应特征值大小")
    # plt.xlabel("特征向量序号", fontsize=17)
    # plt.ylabel("特征值", fontsize=17)
    # # plt.xlim(0.9, 6.1)  # 设置x轴的范围
    # # plt.ylim(1.5, 16)
    # plt.grid()  # 显示网格
    # plt.tick_params(labelsize=15)
    # plt.tight_layout()
    # plt.savefig(FIGURE_PATH + "各特征向量对应特征值大小.png", bbox_inches="tight")
    # plt.show()  # 显示图形

    # 求特征值的贡献度
    contrib = featValue / np.sum(featValue)
    # 求特征值的累计贡献度
    accu_contrib = np.cumsum(contrib)

    # plt.figure(figsize=(6, 4))
    # plt.bar(
    #     range(len(contrib)), contrib, alpha=0.5, align="center", label="贡献度",
    # )
    # plt.step(
    #     range(len(accu_contrib)), accu_contrib, where="mid", label="累积贡献度",
    # )
    # plt.ylabel("贡献度", fontsize=17)
    # plt.xlabel("主成分向量", fontsize=17)
    # plt.tick_params(labelsize=15)
    # plt.legend(loc="best", fontsize=15)
    # plt.tight_layout()
    # plt.savefig(FIGURE_PATH + "各特征向量贡献度.png", bbox_inches="tight")
    # plt.show()

    # 选出主成分 # TODO 主成分精度
    k = [i for i in range(len(accu_contrib)) if accu_contrib[i] < 0.7]

    # 选出主成分对应的特征向量
    selectVec = np.matrix(featVec.T[k]).T
    # 主成分得分，即投影到新坐标下的数据
    finalData = np.dot(feature_array, selectVec)

    # # 绘制热力图
    # fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    # # ax = sns.heatmap(np.abs(selectVec), annot=True, cmap="BuPu")
    # ax = sns.heatmap(selectVec, annot=True, cmap="bwr", annot_kws={"fontsize": 11})
    # ax.set_yticklabels(FEATURE_LIST)
    # # plt.title("Factor Analysis", fontsize="xx-large")
    # plt.yticks(rotation=0, fontsize=13)
    # # plt.ylabel("Sepal Width")
    # plt.tight_layout()
    # # plt.savefig(FIGURE_PATH + "各特征参数对主成分的载荷.png", bbox_inches="tight")
    # plt.show()

    # 原始各特征对于主成分向量的载荷
    contri_rawdata = np.sum(np.abs(selectVec), axis=1)
    contri = np.array(contri_rawdata).squeeze()
    important_feature = []
    important_feature_contri = []
    for index in np.argsort(contri)[-6::]:
        important_feature.append(FEATURE_LIST[index])
        important_feature_contri.append(contri[index])
        print(f"Factor: {contri[index]:03f}\tFeature: {FEATURE_LIST[index]}")

    # fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    # ax = sns.heatmap(
    #     contri_rawdata, annot=True, cmap="Blues", annot_kws={"fontsize": 11}
    # )
    # ax.set_yticklabels(FEATURE_LIST)
    # plt.yticks(rotation=0, fontsize=13)
    # plt.tight_layout()
    # # plt.savefig(FIGURE_PATH + "各特征参数对主成分的累积载荷.png", bbox_inches="tight")
    # plt.show()

    # pca = PCA(n_components=7)
    # newX = pca.fit_transform(np.array(feature_df))  # 等价于pca.fit(X) pca.transform(X)
    # invX = pca.inverse_transform(newX)  # 将降维后的数据转换成原始数据
    # print(pca.explained_variance_ratio_)

    sse = []  # 存放每次结果的误差平方和
    sil_score = []
    test_range = range(2, 12)
    for k in test_range:
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(feature_df[important_feature])
        sse.append(estimator.inertia_)
        sil_score.append(
            silhouette_score(feature_df[important_feature], estimator.labels_)
        )
    plt.xlabel("聚类数量", fontsize=17)
    plt.ylabel("SSE", fontsize=17)
    plt.plot(test_range, sse, "o-")
    # plt.grid()  # 显示网格
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    # plt.savefig(FIGURE_PATH + "聚类SSE.png", bbox_inches="tight")
    plt.show()

    plt.xlabel("聚类数量", fontsize=17)
    plt.ylabel("轮廓系数", fontsize=17)
    plt.plot(test_range, sil_score, "o-")
    # plt.grid()  # 显示网格
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    # plt.savefig(FIGURE_PATH + "聚类轮廓系数.png", bbox_inches="tight")
    plt.show()

    # # 轮廓系数法
    # tests = [2, 3, 4, 5, 8]
    # subplot_counter = 1
    # for t in tests:
    #     subplot_counter += 1
    #     plt.subplot(3, 2, subplot_counter)
    #     kmeans_model = KMeans(n_clusters=t).fit(X)
    #     for i, l in enumerate(kmeans_model.labels_):
    #         plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
    #         # 每个点对应的标签值
    #         # print(kmeans_model.labels_)
    #         plt.xlim([0, 10])
    #         plt.ylim([0, 10])
    #         plt.title('K = %s, Silhouette Coefficient = %.03f' % (t, metrics.silhouette_score(X, kmeans_model.labels_, metric='euclidean')))
