# coding=utf-8
import numpy as np
from numpy import ndarray
from numpy.matrixlib.defmatrix import matrix
from japp.irt.exceptions import SlopThresholdDimError, ThetaItemPareDimError, ItemParaShapeError, SlopShapeError
from japp.utils import cached_property


class LogisticModel(object):
    """
    双参数项目反应理论的基础公式， e^(a*(0-b)) / 1+e^(a*(0-b))
    在这个基础模型中，没有猜测参数，区分度参数必须为一维，难度可以为多维，被试可以为多维
    """

    def __init__(self, slop, threshold, theta):

        """
        生成logistic的原始值和导数值（一阶，二阶，三阶随意）
        :param slop: 斜率，试题区分度，可以是整数、浮点数或numpy数组
        :param threshold: 阈值，试题难度，可以是整数、浮点数或numpy数组
        :param theta: 特质值，浮点或整数，或shape为（XX，1）的numpy二维数组
        """
        if not isinstance(slop, (int, float, list, ndarray, matrix)):
            raise TypeError(u'项目区分度值必须为整数、小数、列表或numpy中的ndarray和matrix')
        if not isinstance(threshold, (int, float, list, ndarray, matrix)):
            raise TypeError(u'项目难度必须为整数、小数、列表或numpy中的ndarray和matrix')
        if not isinstance(theta, (int, float, list, ndarray, matrix)):
            raise TypeError(u'被试特质值必须为整数、小数、列表或numpy中的ndarray和matrix')

        # 统一把参数均转化为numpy的ndarray类型
        if isinstance(slop, (int, float, list, matrix)):
            slop = np.array(slop)
        if isinstance(threshold, (int, float, list, matrix)):
            threshold = np.array(threshold)
        if isinstance(theta, (int, float, list, matrix)):
            theta = np.array(theta)

        threshold_shape = np.shape(slop)
        slop_shape = np.shape(slop)

        # 区分度和难度参数维度的检验，如果是多维数组，则报错
        if len(slop_shape) > 2 or len(threshold_shape) > 2:
            raise ItemParaShapeError(u'区分度参数或难度参数只能是一维数组或二维数组的')
        # 如果区分度和难度的参数个数不对等，则报错
        if slop_shape[0] != np.shape(threshold)[0]:
            raise ItemParaShapeError(u'区分度和难度参数的数量级不匹配')
        # 如果区分度为二维数组，且shape属性不是形如(xx, 1)
        if len(slop_shape) == 2 and slop_shape[1] != 1:
            raise SlopShapeError(u'区分度参数为二维数组时，shape必须形如（xx, 1)')
        # 如果区分度为二维数组，但是难度是一位数组，则区分度降维
        if len(slop_shape) == 2 and len(threshold_shape) == 1:
            slop = slop[0, ]
        # 如果区分度是一位数组，但是难度是二维数组，则区分度升维
        if len(slop_shape) == 1 and len(threshold_shape) == 2:
            slop.shape = slop_shape, 1
        # 如果试题参数的个数大于1，并且被试特质测试也大于1，则报错
        if slop_shape[0] > 1 and np.shape(theta) > 1:
            raise ThetaItemPareDimError(u'试题参数为非标量或包含元素多于1个时，'
                                        u'被试特质参数必须为标量或包含元素仅为1个，反之同理')

        self.slop = slop
        self.threshold = threshold
        self.theta = theta

    @cached_property
    def prob_values(self):
        """
        logistic模型
        e^x/(1+e^x)
        :return: logistic的值
        """
        exp = np.exp(self.slop * (self.theta - self.threshold))
        return exp / (1.0 + exp)

    @cached_property
    def d_prob_values(self):
        """
        logistic的一阶导数
        :return: logistic的一阶导数值
        """
        p = self.prob_values
        return self.slop * p * (1.0 - p)

    @property
    def dd_prob_values(self):

        """
        logistic二阶导数
        :return: logistic二阶导数的值
        """
        dp = self.d_prob_values
        return self.slop * dp