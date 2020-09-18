"""LAB 1."""

from math import sqrt
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
        rand_int = randint(10000000,99999999)

        i = 0
        while i < self.n:
            rand_int = rand_int ** 2
            if len(str(rand_int)) < 16:
                rand_int = "0" * (16 - len(str(rand_int))) + str(rand_int)
            else:
                rand_int = str(rand_int)

            self.result.append(int(rand_int[3:11]) / 100000000)
            rand_int = int(rand_int[3:11])

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
    def solve(self, z: list, method) -> None:
        """"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """"""
        return self.__class__.__name__


class UniformityTest(BaseTest):
    """Uniformity test class"""
    def __init__(self, k: int):
        self.k: int = k
        self.temp_dict: dict = {}
        self.array: list = []

    def build_plot(self, method) -> None:
        """"""
        plt.bar(self.array, list(self.temp_dict.values()), width=0.1, align='edge')
        plt.title(method.__class__.__name__)
        plt.show()

    def solve(self, z: list, method) -> None:
        """"""
        self.array = [i * 1/self.k for i in range(0,11)]
        for index, segment in enumerate(self.array):
            if index + 1 == len(self.array):
                break
            segment = tuple([segment, self.array[index + 1]])
            self.temp_dict[segment] = len([i for i in z if i >= self.array[index] and i < self.array[index + 1]]) / len(z)

        del self.array[-1]

        self.build_plot(method)

    def __str__(self) -> str:
        """"""
        super().__str__()


class IndependenceTest(BaseTest):
    """Independence test class"""
    def __init__(self, k: int, s: int):
        self.k: int = k
        self.s: int = s

    def shift(self, z: list) -> list:
        """"""
        for i in range(self.s):
            z.insert(0, z.pop())
        return z

    def M(self, z: list) -> float:
        """"""
        return 1/len(z) * sum(z)

    def D(self, z: list) -> float:
        """"""
        #return 1/len(z) * sum([i ** 2 - self.M(z) ** 2 for i in z])
        return 1/len(z) * sum(map(lambda x: x ** 2, z)) - self.M(z) ** 2

    def M_x_y(self, z: list, z_shift: list) -> float:
        """"""
        return sum([i * j for i, j in zip(z,z_shift)]) * 1/len(z)

    def corrcoef(self, z: list, z_shift: list) -> float:
        """"""
        return (self.M_x_y(z, z_shift) - self.M(z) * self.M(z_shift)) / sqrt(self.D(z) * self.D(z_shift))

    def solve(self, z: list, method) -> None:
        """"""
        z_shift = self.shift(z.copy())
        print(self.corrcoef(z, z_shift))

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
            z = method.start()
            print(z)
            for test in self.tests:
                test.solve(z, method)


def main() -> None:
    """Main"""
    midsquare_method = MidSquareMethod(10)
    multiplicative_congruent_method = MultiplicativeCongruentMethod(10, 16807, 2 ** 31 - 1)

    independence_test = IndependenceTest(10, 5)
    uniformity_test= UniformityTest(10)

    lab = LAB(midsquare_method, multiplicative_congruent_method)
    lab.add_tests(independence_test, uniformity_test)
    lab.run()


if __name__ == "__main__":
    """Entry point"""
    main()