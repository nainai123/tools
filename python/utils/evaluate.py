from sklearn.metrics import brier_score_loss, confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score, auc, roc_curve, plot_roc_curve, plot_precision_recall_curve, precision_recall_curve,\
    average_precision_score
from matplotlib import pyplot as plt


def brier_score(y_true, y_prob):
    '''

    :param y_true:标签
    :param y_prob: 预测分数
    :return: brier得分
    '''
    return brier_score_loss(y_true, y_prob)


def confusion_matrix(y_true, y_pred):
    '''

    :param y_true: 标签
    :param y_pred: 预测分类
    :return: 混淆矩阵
    '''
    return confusion_matrix(y_true, y_pred)


def acc(y_true, y_pred):
    '''

    :param y_true: 标签
    :param y_pred: 预测分类
    :return: accuracy
    '''
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    '''

    :param y_true: 标签
    :param y_pred: 预测分类
    :return: precision
    '''
    return precision_score(y_true, y_pred)


def recall(y_true, y_pred):
    '''

    :param y_true: 标签
    :param y_pred: 预测分类
    :return: recall
    '''
    return recall_score(y_true, y_pred)


def f1(y_true, y_pred):
    '''

    :param y_true: 标签
    :param y_pred: 预测分类
    :return: f1
    '''
    return  f1_score(y_true, y_pred)


def auc_score(y_true, y_score, pos_label=1):
    '''

    :param y_true: 标签
    :param y_score: 预测分数
    :param pos_label: 阳性标签，默认值为1
    :return:
    '''
    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=pos_label)
    return auc(fpr, tpr)


def draw_roc_curve(y_true, y_score, outpath, pos_label=1):
    '''
    如果分类器不是sklearn的estimator实例，或者预测分数和标签已知，可以使用此函数画roc曲线。
    :param y_true: 标签
    :param y_score: 预测分数
    :param outpath: 输出路径
    :param pos_label: 阳性标签，默认值为1
    :return:
    '''
    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=pos_label)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(outpath)


def draw_pr_curve(y_true, y_score, outpath, pos_label=1):
    """
    如果分类器不是sklearn的estimator实例，或者预测分数和标签已知，可以使用此函数画PR曲线。
    :param y_true: 标签
    :param y_score: 预测分数
    :param outpath: 输出路径
    :param pos_label: 阳性标签，默认值为1
    :return:
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    average_precision = average_precision_score(y_true, y_score, pos_label=pos_label)
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color="darkorange", lw=lw, label="PR curve (area = %0.2f)" % (average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(outpath)


def draw_roc_curve_with_sklearn(classifier, X, y, outpath, pos_label=1):
    """
    使用sklearn内置的画roc曲线的函数，如果分类器是sklearn的estimator实例，推荐使用这个
    :param classifier: 经过训练的分类器实例
    :param X: 测试集
    :param y: 测试集标签
    :param outpath: 图片输出路径
    :return: None
    """
    plot_roc_curve(classifier, X, y, pos_label=pos_label)
    plt.savefig(outpath)


def draw_pr_curve_with_sklearn(classifier, X, y, outpath, pos_label=1):
    """
    使用sklearn内置的画PR曲线的函数，如果分类器是sklearn的estimator实例，推荐使用这个
    :param classifier: 经过训练的分类器实例
    :param X: 测试集
    :param y: 测试集标签
    :param outpath: 图片输出路径
    :return: None
    """
    plot_precision_recall_curve(classifier, X, y, outpath, pos_label=pos_label)
    plt.savefig(outpath)