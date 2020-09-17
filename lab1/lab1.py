"""LAB 1."""


from abc import ABC, abstractmethod


class NotInheritedFromBaseClass(Exception):
    def __init__(self, text):
        self.txt = text


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

    def build_plot(self) -> None:
        """"""
        print("UniformityTest")

    def __str__(self) -> str:
        """"""
        super().__str__()


class IndependenceTest(BaseTest):
    """Independence test class"""
    def __init__(self):
        pass

    def build_plot(self) -> None:
        """"""
        print("IndependenceTest")

    def __str__(self) -> str:
        """"""
        super().__str__()


class BaseMethod(ABC):
    """Absatract base class for method class"""
    @abstractmethod
    def solve(self) -> None:
        """"""
        pass

    @abstractmethod
    def start(self) -> None:
        """"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """"""
        return self.__class__.__name__


class MidSquareMethod(BaseMethod):
    """Mid-square method class"""
    def __init__(self):
        pass

    def solve(self) -> None:
        """"""
        pass

    def start(self) -> None:
        """"""
        print("MidSquareMethod")

    def __str__(self) -> str:
        """"""
        super().__str__()


class MultiplicativeCongruentMethod(BaseMethod):
    """Multiplicative congruent method class"""
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
        super().__str__()


class LAB:
    """Construction and research of the characteristics  of sensors."""
    def __init__(self, *method_objects: tuple) -> None:
        self.methods: list = list(method_objects)

    def add_tests(self, *test_objects: tuple) -> None:
        """"""
        self._tests: list = list(test_objects)

    def _checking_inheritance(self, list_object: list, base_class) -> None:
        """"""
        try:
            for index, obj in enumerate(list_object):
                if not isinstance(obj, base_class):
                    error_text = "{0} not inherited from {1} abstract class".format(obj, base_class)
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
            method.start()
        for test in self.tests:
            test.build_plot()


def main() -> None:
    """Main"""
    midsquare_method = MidSquareMethod()
    multiplicative_congruent_method = MultiplicativeCongruentMethod()

    independence_test = IndependenceTest()
    uniformity_test= UniformityTest()

    lab = LAB(midsquare_method, multiplicative_congruent_method)
    lab.add_tests(independence_test, uniformity_test)
    lab.run()


if __name__ == "__main__":
    """Entry point"""
    main()