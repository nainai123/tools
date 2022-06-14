from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import resample


def random_split(data, label= None, train_ratio=0.7,test_ratio=0.3):
    if label is None:
        train_data, test_data = train_test_split(data,train_size=train_ratio, test_size=test_ratio)
        return train_data, test_data
    else:
        x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=train_ratio, test_size=test_ratio)
        return x_train, x_test, y_train, y_test


def kfold_split(data, label=None, split=10):
    kf = KFold(n_splits=split)
    train_datas = []
    test_datas = []
    if label is not None:
        train_ys = []
        test_ys = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = data.iloc[train_index,:], data.iloc[test_index,:]
        train_datas.append(train_data)
        test_datas.append(test_data)
        if label is not None:
            train_y, test_y = label.iloc[train_index], label.iloc[test_index]
            train_ys.append(train_y)
            test_ys.append(test_y)

    if label is not None:
        return train_datas, test_datas, train_ys, test_ys
    else:
        return train_datas, test_datas


def bootstrap(X, y, n_samples, frequency):
    bs_X = []
    bs_y = []
    for i in range(frequency):
        bs_X.append(resample(X, n_samples=n_samples))
        bs_y.append(resample(y, n_samples=n_samples))
    return bs_X, bs_y





