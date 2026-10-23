from abc import ABC, abstractmethod
from functools import partial, update_wrapper

class Decorator(ABC):
    def __init__(self, f):
        self.func = f
        # updated=[] so that 'self' attributes are not overwritten
        update_wrapper(self, f, updated=[])

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __get__(self, instance, owner):
        new_f = partial(self.__call__, instance)
        update_wrapper(new_f, self.func)
        return new_f
