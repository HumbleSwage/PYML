import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 全局变量
real_weights = 1
real_bias = 3


def generating_label_data(weight, bias, size=10) -> pd.DataFrame:
    """
    产生需要的数据
    数据集必须是线形可分的，而不是随意的数据，具体的流程如下：
    利用高斯白噪声生成基于某个直线附近的若干点：x2=wx1+b
    :return:df
    """
    x1_point = np.linspace(-1, 1, size)[:, np.newaxis]
    noise = np.random.normal(0, 0.5, x1_point.shape)
    x2_point = weight * x1_point + bias + noise
    input_arr = np.hstack((x1_point, x2_point))
    # np.sign(x) 就是符号函数
    label = np.sign(input_arr[:, 1] - (input_arr[:, 0] * real_weights + real_bias)).reshape((size, 1))
    label_data = np.hstack((input_arr, label))
    # 转换为dataFrame
    df = pd.DataFrame(label_data, columns=["x1", "x2", "y"])
    return df


def split_data(data, ratio) -> (pd.DataFrame, pd.DataFrame):
    """
    数据分割
    :param data: 生成的原始整体数据
    :param ratio: 测试数据的比例
    :return:
            train_data指训练数据集
            test_data指测试数据集合
    """
    test_size = int(len(data) * ratio)
    test_data = data.loc[0:test_size, ]
    train_data = data.loc[test_size:, ]
    return train_data, test_data


def original_perceptron(x1_train, x2_train, y_train, x1_test, x2_test, y_test, learn_rate, train_num):
    """感知机原始算法流程如下所示
    输入：训练数据集、学习率
    输出：感知机模型
    1、选取模型的初始值
    2、在训练数据集中选取数据
    3、计算损失函数，如果小于0，则按照指定的策略对参数模型参数进行更新，直到针对所有点的计算损失函数都大于0
    """
    # 初始化w，b
    weight = np.random.rand(2, 1)
    bias = 0
    for rounds in range(train_num):
        for i in range(len(x1_train)):
            # 算法核心：参数的迭代逻辑[这个地方的标注的y和坐标的x和y一定要分开]
            if y_train.loc[i] * (weight[0] * x1_train[i] + weight[1] * x2_train[i] + bias) <= 0:
                weight[0] = weight[0] + learn_rate * y_train[i] * x1_train[i]
                weight[1] = weight[1] + learn_rate * y_train[i] * x2_train[i]
                bias = bias + learn_rate * y_train[i]
        if rounds % 10 == 0:
            learn_rate *= 0.9
            compute_accuracy_callback_f1(x1_test, x2_test, y_test, weight, bias)
    return weight, bias


def compute_accuracy_callback_f1(x1_test, x2_test, y_test, weight, bias):
    """
    计算精度：选择精度、召回率、F1
    :return:
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(x1_test)):
        if y_test[i] != np.sign(x1_test[i] * weight[0] + x2_test[i] * weight[1] + bias):
            if y_test[i] > 0:
                fn += 1
            else:
                fp += 1
        else:
            if y_test[i] > 0:
                tp += 1
            else:
                tn += 1
    if tp + fp == 0 & tp + fn == 0:
        return
    accuracy = tp / (tp + fp)
    callback = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    print("accuracy={0}\t\t\tcallback={1}\t\t\tf1={2}".format(round(accuracy, 5), round(callback, 5), round(f1, 5)))


def data_factory(data):
    """
    返回训练与预测的数据集合
    :param data:原始整体数据集合
    :return:
    """
    train_data, test_data = split_data(data, 0.3)
    x1_train = train_data["x1"].reset_index(drop=True)
    x2_train = train_data["x2"].reset_index(drop=True)
    y_train = train_data["y"].reset_index(drop=True)
    x1_test = test_data["x1"].reset_index(drop=True)
    x2_test = test_data["x2"].reset_index(drop=True)
    y_test = test_data["y"].reset_index(drop=True)
    return x1_train, x2_train, y_train, x1_test, x2_test, y_test


def main():
    size = 50  # 生成的总的数据集个数
    learn_rate = 1  # 学习率
    train_num = 100  # 训练次数
    # 生成数据
    data = generating_label_data(real_weights, real_bias, size)
    # 获取数据
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = data_factory(data)
    # 调用模型
    original_perceptron(x1_train, x2_train, y_train, x1_test, x2_test, y_test, learn_rate, train_num)
    # 作图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(x1_train)):
        if y_train.loc[i] == 1:
            ax.scatter(x1_train[i], x2_train[i], color='r')
        else:
            ax.scatter(x1_train[i], x2_train[i], color='b')
    x = np.linspace(-1, 1.5, 10)
    y1 = real_weights * x + real_bias
    ax.plot(x, y1, color='g')
    plt.show()


if __name__ == '__main__':
    main()
