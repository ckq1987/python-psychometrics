# coding=utf-8


class ItemShapeError(Exception):
    """
    试题区分度数组和难度数组的不一致
    不一致的情况例如
    试题区分度数组, array([1, 2, 3])
    试题难度数组, array([1, 2])
    """


class SlopShapeError(Exception):
    """
    试题区分度数组的shapes属性异常, shape属性应为(x,)或(x, 1)
    """


class ThetaItemPareDimError(Exception):
    """
    试题参数数组与被试特质数组shape的冲突
    当试题参数数组包含多个元素时,特质参数数组仅能包含一个元素
    反之亦然
    """