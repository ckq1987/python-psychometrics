# coding=utf-8
from japp.irt import LogisticModel
from unittest import TestCase
from japp.irt.exceptions import ThetaItemPareDimError, ItemShapeError, SlopShapeError
import numpy as np


class TestLogisticModel(TestCase):

    def test_illegal_args(self):
        """
        测试非法类型的参数
        """
        kwargs = {'slop': 'a', 'threshold': 1, 'theta': 1}
        self.assertRaisesRegexp(TypeError, '区分度必须为整数、小数、列表或numpy中的ndarray和matrix',
                                LogisticModel, **kwargs)
        kwargs = {'slop': 1, 'threshold': 'a', 'theta': 1}
        self.assertRaisesRegexp(TypeError, '难度必须为整数、小数、列表或numpy中的ndarray和matrix',
                                LogisticModel, **kwargs)
        kwargs = {'slop': 1, 'threshold': 1, 'theta': 'a'}
        self.assertRaisesRegexp(TypeError, '特质必须为整数、小数、列表或numpy中的ndarray和matrix',
                                LogisticModel, **kwargs)

    def test_more_than_2_dim_error(self):
        """
        测试参数维度大于2的情况
        """
        kwargs = {'slop': [[[1, 2, 3]]], 'threshold': [1, 2, 3], 'theta': 1}
        self.assertRaisesRegexp(ItemShapeError, '区分度参数或难度参数或特质参数只能是一维数组或二维数组的',
                                LogisticModel, **kwargs)
        kwargs = {'slop': [1, 2, 3], 'threshold': [[[1, 2, 3]]], 'theta': 1}
        self.assertRaisesRegexp(ItemShapeError, '区分度参数或难度参数或特质参数只能是一维数组或二维数组的',
                                LogisticModel, **kwargs)
        kwargs = {'slop': [1, 2, 3], 'threshold': [1, 2, 3], 'theta': [[[1, 2, 3]]]}
        self.assertRaisesRegexp(ItemShapeError, '区分度参数或难度参数或特质参数只能是一维数组或二维数组的',
                                LogisticModel, **kwargs)

    def test_slop_dim_error(self):
        """
        测试区分度数组的维度错误
        """
        kwargs = {'slop': [[1, 2, 3], [1, 2, 3], [1, 2, 3]], 'threshold': [1, 2, 3], 'theta': 1}
        self.assertRaisesRegexp(SlopShapeError, '区分度参数为二维数组时，shape必须形如（xx, 1）或（1， xx）',
                                LogisticModel, **kwargs)

    def test_item_para_dim_error_args(self):
        """
        测试试题区分度的维度和试题难度维度不一致
        """
        kwargs = {'slop': [1, 2, 3], 'threshold': [1, 2], 'theta': 1}
        self.assertRaisesRegexp(ItemShapeError, '区分度和难度参数的数量级不匹配',
                                LogisticModel, **kwargs)
        kwargs = {'slop': [1, 2, 3], 'threshold': [[1, 2, 3]], 'theta': 1}
        self.assertRaisesRegexp(ItemShapeError, '区分度和难度参数的数量级不匹配',
                                LogisticModel, **kwargs)

    def test_theta_dim_error_args(self):
        """
        测试试题参数的维度和被试特质参数维度的冲突
        """
        kwargs = {'slop': [1, 2, 3], 'threshold': [1, 2, 3], 'theta': [[1, 2]]}
        self.assertRaisesRegexp(ThetaItemPareDimError, '被试特质参数为二维数组时，shape属性必须形为（xx, 1）',
                                LogisticModel, **kwargs)

    def test_dtype_error(self):
        """
        测试数组中出现非整数和浮点数的错误
        """
        kwargs = {'slop': ['1', 2.1, 3], 'threshold': [1.2, 2, 3], 'theta': 1}
        self.assertRaisesRegexp(TypeError, "区分度数组或列表或矩阵中存在非整数或非浮点数的数据，请检查",
                                LogisticModel, **kwargs)
        kwargs = {'slop': [1, 2.1, 3], 'threshold': ['1', 2.1, 3], 'theta': 1}
        self.assertRaisesRegexp(TypeError, "难度数组或列表或矩阵中存在非整数或非浮点数的数据，请检查",
                                LogisticModel, **kwargs)
        kwargs = {'slop': [1, 2.1, 3], 'threshold': [1, 2.1, 3], 'theta': [1.2, 2, '3']}
        self.assertRaisesRegexp(TypeError, "特质数组或列表或矩阵中存在非整数或非浮点数的数据，请检查",
                                LogisticModel, **kwargs)

    def test_para_convert_type(self):
        """
        测试参数的转换，int，float，list、matrix均转换为array
        """
        kwargs = {'slop': [1, 2.1, 3.01], 'threshold': [1, 2.01, 3.01], 'theta': 1.0}
        logit_model = LogisticModel(**kwargs)
        np.testing.assert_equal(logit_model.slop, np.array([1, 2.1, 3.01]))
        np.testing.assert_equal(logit_model.threshold, np.array([1, 2.01, 3.01]))
        np.testing.assert_equal(logit_model.theta, np.array([1.0]))
        kwargs = {'slop': [1, 2.1], 'threshold': [[1, 2.01, 3.01], [1, 2.01, 3.01]], 'theta': 1.0}
        logit_model = LogisticModel(**kwargs)
        np.testing.assert_equal(logit_model.slop, np.array([[1], [2.1]]))
        kwargs = {'slop': [[1, 2.1, 3.1]], 'threshold': [1, 2.01, 3.01], 'theta': 1.0}
        logit_model = LogisticModel(**kwargs)
        np.testing.assert_equal(logit_model.slop, np.array([1, 2.1, 3.1]))
        kwargs = {'slop': [[1], [2.1], [3.1]], 'threshold': [1, 2.01, 3.01], 'theta': 1.0}
        logit_model = LogisticModel(**kwargs)
        np.testing.assert_equal(logit_model.slop, np.array([1, 2.1, 3.1]))

    def test_matrix_para_convert_type(self):
        kwargs = {'slop': np.matrix([1, 0.1, 0.2]), 'threshold': [1, 2.01, 3.01], 'theta': 1.0}
        logit_model = LogisticModel(**kwargs)
        np.testing.assert_equal(logit_model.slop, np.array([1, 0.1, 0.2]))