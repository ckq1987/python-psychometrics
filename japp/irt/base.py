# coding=utf-8
import numpy as np
from japp.irt.exceptions import ThetaItemPareDimError, ItemShapeError, SlopShapeError
from japp.utils import cached_property
from japp.utils.tools import base_item_parameter_type_check


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
        slop = base_item_parameter_type_check(slop, '区分度')
        threshold = base_item_parameter_type_check(threshold, '难度')
        theta = base_item_parameter_type_check(theta, '特质')

        slop, theta = self.__shape_check(slop, threshold, theta)

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

    @staticmethod
    def __shape_check(slop, threshold, theta):
        # TODO 下面代码太糟糕了，待有空重构下
        """
        项目参数和特质参数的维度检验
        并对slop和theta进行维度转换
        :param slop: numpy数组
        :param theta: numpy数组
        :param threshold: numpy数组
        :raise ThetaItemPareDimError:
        """

        # 下面是参数的维度形状
        threshold_shape = np.shape(threshold)
        slop_shape = np.shape(slop)
        theta_shape = np.shape(theta)

        # 下面是参数的维度个数
        slop_dim_count = len(slop_shape)
        threshold_dim_count = len(threshold_shape)
        theta_dim_count = len(theta_shape)

        # 区分度和难度参数维度的检验，如果是多维数组，则报错
        if slop_dim_count > 2 or threshold_dim_count > 2 or theta_dim_count > 2:
            raise ItemShapeError('区分度参数或难度参数或特质参数只能是一维数组或二维数组的')
        else:
            # 如果区分度和难度的参数个数不对等，则报错
            if slop_dim_count == 1:
                if slop_shape[0] != threshold_shape[0]:
                    raise ItemShapeError('区分度和难度参数的数量级不匹配')
                # 如果区分度是一位数组，但是难度是二维数组，则区分度升维
                if threshold_dim_count == 2:
                    slop.shape = slop_shape[0], 1
            # 如果区分度为二维数组，且shape属性不是形如(xx, 1)或（1， xx）
            elif slop_dim_count == 2:
                if slop_shape[1] == 1:
                    if slop_shape[0] != threshold_shape[0]:
                        raise ItemShapeError('区分度和难度参数的数量级不匹配')
                    # 如果区分度为二维数组，但是难度是一位数组，则区分度降维
                    if threshold_dim_count == 1:
                        slop.shape = (slop_shape[0],)
                elif slop_shape[0] == 1:
                    if slop_shape[1] != threshold_shape[0]:
                        raise ItemShapeError('区分度和难度参数的数量级不匹配')
                    # 如果区分度为二维数组，但是难度是一位数组，则区分度降维
                    if threshold_dim_count == 1:
                        slop.shape = (slop_shape[1],)
                else:
                    raise SlopShapeError('区分度参数为二维数组时，shape必须形如（xx, 1）或（1， xx）')

            # 如果试题参数的个数大于1，并且被试特质测试也大于1，则报错
            if theta_dim_count == 2:
                if theta_shape[1] != 1:
                    raise ThetaItemPareDimError('被试特质参数为二维数组时，shape属性必须形为（xx, 1）')
            else:
                if theta_shape[0] != 1:
                    theta.shape = theta_shape[0], 1
        return slop, theta