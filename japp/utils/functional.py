# coding=utf-8


class CachedProperty(object):
    """
    缓存属性的装饰器，从django抄的，
    详见https://docs.djangoproject.com/en/1.9/ref/utils/#module-django.utils.functional
    """
    def __init__(self, func, name=None):
        self.func = func
        self.__doc__ = getattr(func, '__doc__')
        self.name = name or func.__name__

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res

cached_property = CachedProperty