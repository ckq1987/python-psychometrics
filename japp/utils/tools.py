# coding=utf-8
import numpy as np
from numpy.matrixlib.defmatrix import matrix


def __dtype_check(array_value, parameter_name):
    """
    对array数组dtype类型进行检查的私有函数
    :param array_value: array数组
    :param parameter_name: 试题参数名字，区分度、难度、特质等等
    :raise TypeError: 数组或列表或矩阵中存在非整数或非浮点数的数据，请检查
    """
    dtype = array_value.dtype
    if not np.issubdtype(dtype, np.integer) and not np.issubdtype(dtype, np.float):
        error_msg = "%s数组或列表或矩阵中存在非整数或非浮点数的数据，请检查" % parameter_name
        raise TypeError(error_msg)


def base_item_parameter_type_check(parameter_value, parameter_name):
    """
    项目参数和特质参数的基本类型检查
    :param parameter_value: 参数值
    :param parameter_name: 参数名, 例如, 区分度, 难度, 特质
    :return: 转换后的参数值, 类型统一是ndarray
    """
    if isinstance(parameter_value, (int, float)):
        parameter_value = np.array([parameter_value])
    elif isinstance(parameter_value, list):
        parameter_value = np.array(parameter_value)
        __dtype_check(parameter_value, parameter_name)
    elif isinstance(parameter_value, np.ndarray):
        __dtype_check(parameter_value, parameter_name)
    elif isinstance(parameter_value, matrix):
        if parameter_value.shape[0] == 1 or parameter_value.shape[1] == 1:
            parameter_value = parameter_value.A1
            __dtype_check(parameter_value, parameter_name)
        else:
            error_msg = "当{0}是matrix类型, 仅允许shape属性为(xx, 1)或 (1, xx)".format(parameter_name)
            raise TypeError(error_msg)
    else:
        error_msg = "{0}必须为整数、小数、列表或numpy中的ndarray和matrix".format(parameter_name)
        raise TypeError(error_msg)
    return parameter_value