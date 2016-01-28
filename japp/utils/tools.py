# coding=utf-8
from numpy import ndarray, array
from numpy.matrixlib.defmatrix import matrix


def base_item_parameter_type_check(parameter_value, parameter_name):
    """
    项目参数和特质参数的基本类型检查
    :param parameter_value: 参数值
    :param parameter_name: 参数名, 例如, 区分度, 难度, 特质
    :return: 转换后的参数值
    """
    if not isinstance(parameter_value, (int, float)):
        parameter_value = array([parameter_value])
    elif isinstance(parameter_value, list):
        parameter_value = array(parameter_value)
    elif isinstance(parameter_value, ndarray):
        parameter_value = parameter_value
    elif isinstance(parameter_value, matrix):
        if parameter_value.shape[0] == 1 or parameter_value.shape[1] == 1:
            parameter_value = parameter_value.A1
        else:
            raise TypeError(u'当%s是matrix类型, 仅允许shape属性为(xx, 1)或 (1, xx)' % parameter_name)
    else:
        raise TypeError(u'%s必须为整数、小数、列表或numpy中的ndarray和matrix' % parameter_name)
    return parameter_value
