import numpy as np

def winsorized_std(data, n=3):
    """
    :param data: 待处理数据
    :param n: 标准差去极值的参数
    """
    mean, std = data.mean(), data.std()
    return data.clip(mean - std * n, mean + std * n)


def winsorized_percentile(data, left=0.025, right=0.975):
    """
    :param data: 待处理数据
    :param left: 百分法去极值的左端
    :param right: 百分法去极值的右端
    """
    lv, rv = np.percentile(data, [left * 100, right * 100])
    return data.clip(lv, rv)


def winsorized_mad(data, n=3):
    """
    :param data: 待处理数据
    :param n: 中位数去极值的参数
    """
    median = data.median()
    median_diff = (data - median).abs().median()
    return data.clip(median - median_diff * n, median + median_diff * n)


def standardize(data):
    """
    :param data: 待处理数据
    """
    return (data-data.mean())/data.std()


