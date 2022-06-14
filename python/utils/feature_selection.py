from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
import heapq
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


def variance_threshold(X, threshold, get_feature_importance=False):
    '''
    方差选择
    使用方差选择法，先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。
    只需修改参数threshold, 将筛选出方差大于threshold的变量。
    这个方法的特征权重用方差大小表示，可以认为方差越大，特征越好
    '''
    model = VarianceThreshold(threshold)
    model.fit(X)
    features = X.columns[model.get_support().tolist()]
    if not get_feature_importance:
        return features
    else:
        variance = model.variances_
        return features, variance


def chi_square(X, y, k, get_feature_importance=False):
    '''
    卡方选择，用于分类任务
    根据特征与目标的独立性选择特征
    只需修改参数k, 表示保留的变量个数
    这个方法的特征权重用卡方值表示，可以认为卡方值越大，特征越好
    '''
    model = SelectKBest(chi2, k)
    model.fit(X, y)
    features = X.columns[model.get_support().tolist()]
    if not get_feature_importance:
        return features
    else:
        chi2_value = model.scores_
        return features, chi2_value


def f_stat(X, y, k, task_type="clf", get_feature_importance=False):
    '''
    F检验（ANOVA选择）,
    参数k, 表示保留的变量个数
    参数task_type，任务类型，可以选择分类（clf)和回归(reg)
    这个方法的特征权重用F值表示，可以认为F值越大，特征越好
    '''
    if task_type == "clf":
        model = SelectKBest(f_classif, k)
        model.fit(X, y)
        selected_features = X.columns[model.get_support().tolist()]

    elif task_type == "reg":
        model = SelectKBest(f_regression, k)
        model.fit(X, y)
        selected_features = X.columns[model.get_support().tolist()]
    else:
        raise Exception("Unknown task type")

    if not get_feature_importance:
        return selected_features
    else:
        f_value = model.scores_
        return selected_features, f_value


def mutual_info(X, y, k, task_type="clf", get_feature_importance=False):
    '''
    互信息法
    参数k, 表示保留的变量个数
    参数task_type，任务类型，可以选择分类（clf)和回归(reg)
    这个方法的特征权重用互信息值表示，可以认为互信息值越大，特征越好
    '''
    if task_type == "clf":
        model = SelectKBest(mutual_info_classif, k)
        model.fit(X, y)
        selected_features = X.columns[model.get_support().tolist()]
    elif task_type == "reg":
        model = SelectKBest(mutual_info_regression, k)
        model.fit(X, y)
        selected_features = X.columns[model.get_support().tolist()]
    else:
        raise Exception("Unknown task type")

    if not get_feature_importance:
        return selected_features
    else:
        mi = model.scores_
        return selected_features, mi


def correlation(X, y, k, method="pearsonr", get_feature_importance=False):
    '''
    相关系数法
    参数method,可以选择“pearsonr”或“spearmanr"
    参数k, 表示保留的变量个数
    这个方法的特征权重用相关系数表示，可以认为相关系数越大，特征越好
    '''
    rs = []
    for i in range(X.shape[1]):
        if method=="pearsonr":
            r, p = pearsonr(X.iloc[:, i], y)
        elif method=="spearmanr":
            r, p = spearmanr(X.iloc[:, i], y)
        else:
            raise Exception("Unrecognized correlation method")
        rs.append(r)
    ind = list(map(rs.index, heapq.nlargest(k, rs)))
    selected_features = X.columns[ind]
    if not get_feature_importance:
        return selected_features
    else:
        coefficients = rs[ind]
        return selected_features, coefficients


def rfe(X, y, k, mode="LR", get_feature_importance=False):
    '''
    Wrapper法
    RFE
    参数k，表示保留的变量个数
    参数mode，wrapper方法中使用的模式，可选
        1. "LR", 默认, logistic regression
        2. "SVC", 线性核SVM
        3. "DT", "decision tree"
        4. "DTR", "decision tree regressor"
        5. "RF", "random forest"
        6. "RFR", "random forest regressor"
        7. "GB", "gradient boosting"
        8. "GBR", "gradient boosting regressor"
          如果为回归任务，应选择DTR，RFR，GBR.各个estimator使用默认参数，要修改estimator的参数，请参考sklearn的官方文档
    '''
    if mode == "LR":
        estimator = LogisticRegression()
    elif mode == "SVC":
        estimator = SVC(kernel="linear", C=1)
    elif mode == "DT":
        estimator = DecisionTreeClassifier()
    elif mode == "RF":
        estimator = RandomForestClassifier()
    elif mode == "GB":
        estimator = GradientBoostingClassifier()
    elif mode == "DTR":
        estimator = DecisionTreeRegressor()
    elif mode == "RFR":
        estimator = RandomForestRegressor()
    elif mode == "GBR":
        estimator = GradientBoostingRegressor()
    else:
        raise Exception("Unrecognized estimator")

    model = RFE(estimator=estimator, n_features_to_select=k)
    model.fit(X, y)
    selected_features = X.columns[model.get_support().tolist()]
    if not get_feature_importance:
        return selected_features
    else:
        estimator = model.estimator_
        if mode in ['LR', 'SVC']:
            feature_importance = estimator.coef_
        elif mode in ['DT', 'DTR', 'RF', 'RFR', 'GB', 'GBR']:
            feature_importance = estimator.feature_importances_
    return selected_features, feature_importance



def rfecv(X, y, mode="LR", n_splits=5, random_state=1, scoring="neg_mean_squared_error"):
    '''
    Wrapper法
    RFECV
    这个方法通过交叉验证自动返回最佳特征，不需要指定返回特征数量，但是消耗时间较长
    参数mode，wrapper方法中使用的模式，可选
        1. "LR", 默认, logistic regression
        2. "SVC", 线性核SVM
        3. "DT", "decision tree"
        4. "DTR", "decision tree regressor"
        5. "RF", "random forest"
        6. "RFR", "random forest regressor"
        7. "GB", "gradient boosting"
        8. "GBR", "gradient boosting regressor"
    如果为回归任务，应选择DTR，RFR，GBR.各个estimator使用默认参数，要修改estimator的参数，请参考sklearn的官方文档
    n_splits, 交叉验证次数
    random_state, 交叉验证切割数据的种子
    scoring, 交叉验证用的评估指标
    '''
    if mode == "LR":
        estimator = LogisticRegression()
    elif mode == "SVC":
        estimator = SVC(kernel="linear", C=1)
    elif mode == "DT":
        estimator = DecisionTreeClassifier()
    elif mode == "RF":
        estimator = RandomForestClassifier()
    elif mode == "GB":
        estimator = GradientBoostingClassifier()
    elif mode == "DTR":
        estimator = DecisionTreeRegressor()
    elif mode == "RFR":
        estimator = RandomForestRegressor()
    elif mode == "GBR":
        estimator = GradientBoostingRegressor()
    else:
        raise Exception("Unrecognized estimator")

    model = RFECV(estimator=estimator, cv=KFold(n_splits=n_splits, random_state=random_state), scoring=scoring)
    model.fit(X, y)
    return X.columns[model.get_support().tolist()]


def embedding(X, y, mode = "LR"):
    '''
     embedding法
     通过训练模型自动选择最佳特征，不需要指定返回特征数量
     参数mode，embedding方法中使用的模式，可选
         1. "LR", 默认, logistic regression
         2. "Lasso"
         3. "RF", "random forest"
         4. "RFR", "random forest regressor"
         5. "GB", "gradient boosting"
         6. "GBR", "gradient boosting regressor"
     如果为回归任务，应选择Lasso, RFR, GBR.各个estimator使用默认参数，要修改estimator的参数，请参考sklearn的官方文档
     '''
    if mode == "LR":
        model = SelectFromModel(LogisticRegression())
    elif mode == "GBR":
        model = SelectFromModel(GradientBoostingRegressor())
    elif mode == "GB":
        model = SelectFromModel(GradientBoostingClassifier())
    elif mode == "RFR":
        model = SelectFromModel(RandomForestRegressor())
    elif mode == "Lasso":
        model = SelectFromModel(Lasso())
    elif mode == "RF":
        model = SelectFromModel(RandomForestClassifier())
    else:
        raise Exception("Unrecognized model")
    model.fit(X, y)
    return X.columns[model.get_support().tolist()]



