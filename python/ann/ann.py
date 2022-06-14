import tensorflow as tf
import pandas as pd
from python.utils import data_split, evaluate
from python.utils import feature_selection
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np


def construct_model(feature_num, structure, output_num):
    hidden_layer = structure.get('hidden_layer')

    l2 = structure.get("l2")
    if l2 is None:
        kernel_regularizer = None
    else:
        kernel_regularizer = tf.keras.regularizers.l2(l2)

    dropouts = structure.get('dropout')
    batch_norm = structure.get('use_batch_norm')
    activation = structure.get('activation')

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(feature_num, )))

    for layer_units, dropout_rate, batch_norm in zip(hidden_layer, dropouts, batch_norm):
        model.add(tf.keras.layers.Dense(layer_units, kernel_regularizer=kernel_regularizer))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
        if dropout_rate:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(output_num))
    if output_num > 1:
        model.add(tf.keras.layers.Activation("softmax"))
    else:
        model.add(tf.keras.layers.Activation("sigmoid"))

    return model


# 读入训练数据, 要求是pandas datframe格式
# data = pd.read_csv("E:/呼研院肺癌/OAK-POPLAR数据-0721.csv", usecols=['Study','age','SLD','METSITES','ORR'])
# 切割数据，执行代码时，选择一种数据分割的方式即可，其他切割方式需要注释掉
# 随机切割，标签在data中
# train_data, test_data = data_split.random_split(data, train_ratio=0.7, test_ratio=0.3)
#
# # 随机切割，标签与data独立,需要将label改为实际的label数据
# x_train, x_test, y_train, y_test = data_split.random_split(data.drop('ORR', axis=1), label=data['ORR'], train_ratio=0.7, test_ratio=0.3)
#
# # kFold切割，标签在data中
# trains, tests = data_split.kfold_split(data, split=10)
#
# # kFold切割，标签与data独立，需要将label改为实际的label数据
# x_trains, x_tests, y_trains, y_tests = data_split.kfold_split(data.drop('ORR', axis=1), label=data['ORR'], split=10)
#
# # bootstrap,标签与data独立，需要指定重复抽样的样本数和抽样次数，抽样结果不区分训练集和测试集
# bs_X, bs_y = data_split.bootstrap(data.drop('ORR', axis=1), y=data['ORR'], n_samples=500, frequency=3)
#
# # 特征选择
# selected_features, feature_weights = feature_selection.rfe(data.drop(['ORR','Study'],axis=1), data['ORR'], k=3, mode="GB", get_feature_importance=True)
#
# # 特征重要性排名图表，写入本地
# # 输出路径
# outpath = "E:/算法模块/python/ann/test.png"
# df = pd.DataFrame({'features':selected_features, 'feature_importances':feature_weights})
# df = df.sort_values('feature_importances', ascending=False)
# sns.barplot(x="feature_importances", y="features", color="tab:blue", data=df)
# plt.savefig(outpath)

iris = datasets.load_iris()
X = iris.data
y = iris.target
y = y[y != 2]
X = X[:len(y)]
train_data, test_data, train_label, test_label = data_split.random_split(X, y, 0.3)
# train_label = data[data.Study == "OAK"]['ORR']
# test_data = data[data.Study == "POPLAR"][['age','SLD','METSITES']]
# test_label = data[data.Study == "POPLAR"]['ORR']


# 设置特征数目
n_inputs = 4
# 设置输出神经元数目
# 1表示二分类模型，若是多分类模型，需要把标签转换为onehot，转换后向量的大小即为n_outputs的值
n_outputs = 1


# 构建网络
# 设置隐藏层数目
hidden_layer = [10,10,10]
# 设置各隐藏层的dropout,用于控制模型的过拟合,如果需要使用dropout,
# 把False改为改为0-1的值即可，优先试用0.5
use_dropout = [False,False,False]
# 设置各隐藏层是否使用batch_norm，用于加快收敛速度和稳定传播梯度
use_batch_norm = [False,False,False]



# l2 regularize 控制过拟合， 默认值为None, 如需要设置，改为数字
l2 = 1
# 激活函数，更多激活函数请参考 https://www.tensorflow.org/api_docs/python/tf/keras/activations
activation = "relu"
# 将参数整合在字典中
structure = {'hidden_layer':hidden_layer, 'dropout':use_dropout, 'use_batch_norm':use_batch_norm, 'l2':l2,
             'activation':activation}

model = construct_model(n_inputs, structure, n_outputs)


# 设置learning_rate
learning_Rate = 0.001
# 设置loss函数
if n_outputs > 1:
    loss = tf.keras.losses.categorical_crossentropy
else:
    loss = tf.keras.losses.binary_crossentropy


model.compile(
    # 更多优化函数请参考 https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_Rate),
    loss=loss,
    # 更多评估指标 https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    metrics=['accuracy']
)

#训练模型，并从中取0.2作为验证(validation_split)，其余参数按实际情况（数据集大小）调整，
# 详细参数说明请参考 https://www.tensorflow.org/api_docs/python/tf/keras/Model
model.fit(train_data, train_label, epochs=100, steps_per_epoch=10, batch_size=32, validation_split=0.2)
#
# 预测
y_score = model.predict(test_data)
loss, acc = model.evaluate(test_data, test_label)

# ROC曲线
test_roc = "E:/算法模块/python/ann/test_prcurve.png"
evaluate.draw_pr_curve(test_label, y_score, test_roc)
print("test loss: %.2f" % (loss))
print("test acc: %.2f" % (acc))




