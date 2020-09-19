"""LAB 1."""


import numpy as np

from math import sqrt
from random import randint, seed
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt


class NotInheritedFromBaseClass(Exception):
    """Ecxeption class"""
    def __init__(self, text):
        self.txt = text


class BaseMethod(ABC):
    """Absatract base class for method class"""
    @abstractmethod
    def solve(self) -> None:
        """Solve method."""
        pass

    @abstractmethod
    def start(self) -> list:
        """Start method."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


class MidSquareMethod(BaseMethod):
    """Mid-square method class"""
    def __init__(self, n: int):
        self.n: int = n
        self.result: list = []

    def solve(self) -> None:
        """Solve mid-square method."""
        rand_int = randint(1000000000000000,9999999999999999)

        i = 0
        while i < self.n:
            rand_int = rand_int ** 2
            if len(str(rand_int)) < 32:
                rand_int = "0" * (32 - len(str(rand_int))) + str(rand_int)
            else:
                rand_int = str(rand_int)

            self.result.append(int(rand_int[7:23]) / 10000000000000000)
            rand_int = int(rand_int[7:23])

            i += 1

    def start(self) -> list:
        """Start solve method."""
        seed(randint(1,100))

        self.solve()

        return self.result

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class MultiplicativeCongruentMethod(BaseMethod):
    """Multiplicative congruent method class"""
    def __init__(self, n: int, k: int, m: int):
        self.n: int = n
        self.k: int = k
        self.m: int = m
        self.result: list = []

    def solve(self) -> None:
        """Solve multiplicative congruent method."""
        rand_int = randint(1000,9999)

        i = 0
        while i < self.n:
            rand_int = (self.k * rand_int) % self.m

            self.result.append(rand_int / self.m)

            i += 1

    def start(self) -> list:
        """Start solve method."""
        seed(randint(100,200))

        self.solve()

        return self.result

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class FuncMixin:
    """Func mixin class with dispersion and checkmate waiting."""
    def M(self, z: list) -> float:
        """Сheckmate waiting M(x)."""
        return 1/len(z) * sum(z)

    def D(self, z: list) -> float:
        """Dispersion D(x)."""
        return 1/len(z) * sum(np.array(z) ** 2 - self.M(z) ** 2)


class BaseTest(ABC):
    """Absatract base class for test class."""
    @abstractmethod
    def test(self, z: list, method) -> None:
        """Test abstract method."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


class UniformityTest(BaseTest, FuncMixin):
    """Uniformity test class."""
    def __init__(self, k: int):
        self.k: int = k
        self.temp_dict: dict = {}
        self.array: list = []

    def M(self, z: list) -> float:
        """Сheckmate waiting M(x)."""
        return super().M(z)

    def D(self, z: list) -> float:
        """Dispersion D(x)."""
        return super().D(z)

    def build_plot(self, method) -> None:
        """Build plot of test result."""
        nums = list(self.temp_dict.values())

        plt.bar(self.array, nums, width=0.1, align='edge')
        plt.title("\n".join([
            method.__class__.__name__,
            f"D = {self.D(method.result)}",
            f"M = {self.M(method.result)}"
            ]))
        plt.show()

    def test(self, z: list, method) -> None:
        """Test method."""
        self.array = [i * 1/self.k for i in range(0,11)]
        for index, segment in enumerate(self.array):
            if index + 1 == len(self.array):
                break
            segment = tuple([segment, self.array[index + 1]])
            self.temp_dict[segment] = len([i for i in z if i >= self.array[index] and i < self.array[index + 1]]) / len(z)

        del self.array[-1]

        self.build_plot(method)

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class IndependenceTest(BaseTest, FuncMixin):
    """Independence test class."""
    def __init__(self, k: int, s: int):
        self.k: int = k
        self.s: int = s

    def shift(self, z: list) -> list:
        """Shift on s positions."""
        for i in range(self.s):
            z.insert(0, z.pop())
        return z

    def M(self, z: list) -> float:
        """Сheckmate waiting M(x)."""
        return super().M(z)

    def D(self, z: list) -> float:
        """Dispersion D(x)."""
        return super().D(z)

    def M_x_y(self, z: list, z_shift: list) -> float:
        """Сheckmate waiting M(xy)"""
        return sum([i * j for i, j in zip(z, z_shift)]) * 1/len(z)

    def corrcoef(self, z: list, z_shift: list) -> None:
        """Calculate and print corr coef."""
        corr_coef = (self.M_x_y(z, z_shift) - self.M(z) * self.M(z_shift)) / sqrt(self.D(z) * self.D(z_shift))
        print("Corr coef = {0}".format(corr_coef))

    def test(self, z: list, method) -> None:
        """Test method."""
        z_shift = self.shift(z.copy())
        self.corrcoef(z, z_shift)

    def __str__(self) -> str:
        """__str__"""
        super().__str__()


class LAB:
    """Construction and research of the characteristics  of sensors."""
    def __init__(self) -> None:
        self.methods: list = []
        self.tests: list = []

    def create_list_of_methods(self, *method_objects: tuple) -> None:
        """Create list of methods."""
        self.methods: list = list(method_objects)

    def create_list_of_tests(self, *test_objects: tuple) -> None:
        """Create list of tests."""
        self.tests: list = list(test_objects)

    def _checking_inheritance(self, list_object: list, base_class) -> None:
        """Check inheritance."""
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
        """Get methods."""
        return self._methods
    
    @methods.setter
    def methods(self, *methods) -> None:
        """Set metods."""
        methods = methods[0]
        self._checking_inheritance(methods, BaseMethod)

    @property
    def tests(self) -> list:
        """Get tests."""
        return self._tests
    
    @tests.setter
    def tests(self, *tests) -> None:
        """Set tests."""
        tests = tests[0]
        self._checking_inheritance(tests, BaseTest)

    def run(self) -> None:
        """Function for run and test all methods"""
        for method in self.methods:
            z = method.start()
            for test in self.tests:
                test.test(z, method)


def main() -> None:
    """Main"""
    midsquare = MidSquareMethod(1000000)
    multiplicative_congruent = MultiplicativeCongruentMethod(1000000, 16807, 2 ** 31 - 1)

    independence = IndependenceTest(10, 5)
    uniformity = UniformityTest(10)

    lab = LAB()
    lab.create_list_of_methods(midsquare, multiplicative_congruent)
    lab.create_list_of_tests(independence, uniformity)
    lab.run()


if __name__ == "__main__":
    """Entry point"""
    main()