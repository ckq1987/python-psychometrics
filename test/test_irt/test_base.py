# coding=utf-8
from japp.irt import LogisticModel
from unittest import TestCase
from japp.irt.exceptions import ThetaItemPareDimError, ItemParaShapeError, SlopShapeError


class TestLogisticModel(TestCase):

    def test_illegal_args(self):
        """
        测试非法类型的参数
        """
        kwargs = {'slop': 'a', 'threshold': 1, 'theta': 1}
        self.assertRaises(TypeError, LogisticModel, **kwargs)
        kwargs = {'slop': 1, 'threshold': 'a', 'theta': 1}
        self.assertRaises(TypeError, LogisticModel, **kwargs)
        kwargs = {'slop': 1, 'threshold': 1, 'theta': 'a'}
        self.assertRaises(TypeError, LogisticModel, **kwargs)

    def test_more_than_2_dim_error(self):
        """
        测试参数维度大于2的情况
        """
        kwargs = {'slop': [[[1, 2, 3]]], 'threshold': [1, 2, 3], 'theta': 1}
        self.assertRaises(ItemParaShapeError, LogisticModel, **kwargs)

    def test_slop_dim_error(self):
        kwargs = {'slop': [[1, 2, 3]], 'threshold': [1, 2, 3], 'theta': 1}
        self.assertRaises(SlopShapeError, LogisticModel, **kwargs)

    def test_item_para_dim_error_args(self):
        """
        测试试题区分度的维度和试题难度维度不一致
        """
        kwargs = {'slop': [1, 2, 3], 'threshold': [1, 2], 'theta': 1}
        self.assertRaises(ItemParaShapeError, LogisticModel, **kwargs)

    def test_theta_dim_error_args(self):
        """
        测试试题参数的维度和被试特质参数维度的冲突
        """
        kwargs = {'slop': [1, 2, 3], 'threshold': [1, 2, 3], 'theta': [1, 2]}
        self.assertRaises(ThetaItemPareDimError, LogisticModel, **kwargs)

    def test_para_convert_type(self):
        """
        测试参数的转换，除了标量，list、matrix均转换为array
        """

