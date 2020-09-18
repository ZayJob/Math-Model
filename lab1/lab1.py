"""LAB 1."""

from random import randint, seed
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt


class NotInheritedFromBaseClass(Exception):
    def __init__(self, text):
        self.txt = text


class BaseMethod(ABC):
    """Absatract base class for method class"""
    @abstractmethod
    def solve(self) -> None:
        """"""
        pass

    @abstractmethod
    def start(self) -> list:
        """"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """"""
        return self.__class__.__name__


class MidSquareMethod(BaseMethod):
    """Mid-square method class"""
    def __init__(self, n: int):
        self.n: int = n
        self.result: list = []

    def solve(self) -> None:
        """"""
        rand_int = randint(1000,9999)

        i = 0
        while i < self.n:
            rand_int = rand_int ** 2
            if len(str(rand_int)) < 8:
                rand_int = "0" * (8 - len(str(rand_int))) + str(rand_int)
            else:
                rand_int = str(rand_int)

            self.result.append(int(rand_int[1:5]))
            rand_int = int(rand_int[1:5])

            i += 1


    def start(self) -> list:
        """"""
        seed(randint(1,100))

        self.solve()

        return self.result

    def __str__(self) -> str:
        """"""
        super().__str__()


class MultiplicativeCongruentMethod(BaseMethod):
    """Multiplicative congruent method class"""
    def __init__(self, n: int, k: int, m: int):
        self.n: int = n
        self.k: int = k
        self.m: int = m
        self.result: list = []

    def solve(self) -> None:
        """"""
        rand_int = randint(1000,9999)

        i = 0
        while i < self.n:
            rand_int = (self.k * rand_int) % self.m

            self.result.append(rand_int / self.m)

            i += 1


    def start(self) -> list:
        """"""
        seed(randint(100,200))

        self.solve()

        return self.result

    def __str__(self) -> str:
        """"""
        super().__str__()


class BaseTest(ABC):
    """Absatract base class for test class"""
    @abstractmethod
    def build_plot(self) -> None:
        """"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """"""
        return self.__class__.__name__


class UniformityTest(BaseTest):
    """Uniformity test class"""
    def __init__(self):
        pass

    def build_plot(self, result: list) -> None:
        """"""
        pass

    def __str__(self) -> str:
        """"""
        super().__str__()


class IndependenceTest(BaseTest):
    """Independence test class"""
    def __init__(self):
        pass

    def build_plot(self, result: list) -> None:
        """"""
        print("IndependenceTest")

    def __str__(self) -> str:
        """"""
        super().__str__()


class LAB:
    """Construction and research of the characteristics  of sensors."""
    def __init__(self, *method_objects: tuple) -> None:
        self.methods: list = list(method_objects)

    def add_tests(self, *test_objects: tuple) -> None:
        """"""
        self.tests: list = list(test_objects)

    def _checking_inheritance(self, list_object: list, base_class) -> None:
        """"""
        try:
            for index, obj in enumerate(list_object):
                if not isinstance(obj, base_class):
                    error_text = "{0} not inherited from {1} abstract class".format(obj, base_class.__name__)
                    del list_object[index]
                    raise NotInheritedFromBaseClass(error_text)
                if base_class is BaseMethod:
                    self._methods = list_object
                else:
                    self._tests = list_object
        except Exception as ex:
            print(ex)

    @property
    def methods(self) -> list:
        """"""
        return self._methods
    
    @methods.setter
    def methods(self, *methods) -> None:
        """"""
        methods = methods[0]
        self._checking_inheritance(methods, BaseMethod)


    @property
    def tests(self) -> list:
        """"""
        return self._tests
    
    @tests.setter
    def tests(self, *tests) -> None:
        """"""
        tests = tests[0]
        self._checking_inheritance(tests, BaseTest)

    
    def run(self) -> None:
        """Function for run all methods"""
        for method in self.methods:
            result = method.start()
            for test in self.tests:
                test.build_plot(result)

# TEST
class TestRaise():
    """TestRaise class"""
    def __init__(self):
        pass

    def solve(self) -> None:
        """"""
        pass

    def start(self) -> None:
        """"""
        print("MultiplicativeCongruentMethod")

    def __str__(self) -> str:
        """"""
        return self.__class__.__name__


def main() -> None:
    """Main"""
    midsquare_method = MidSquareMethod(5)
    multiplicative_congruent_method = MultiplicativeCongruentMethod(20, 7, 13)
    test_raise = TestRaise()

    independence_test = IndependenceTest()
    uniformity_test= UniformityTest()

    lab = LAB(midsquare_method, multiplicative_congruent_method, test_raise)
    lab.add_tests(independence_test, uniformity_test)
    lab.run()


if __name__ == "__main__":
    """Entry point"""
    main()