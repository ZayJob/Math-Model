"""LAB 1."""


from abc import ABC, abstractmethod


class NotInheritedFromBaseMethod(Exception):
    def __init__(self, text):
        self.txt = text


class BaseMethod(ABC):
    """Absatract base class for method class"""
    def __init__(self):
        pass

    @abstractmethod
    def solve(self) -> None:
        """"""
        pass

    @abstractmethod
    def run(self) -> None:
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

    def run(self) -> None:
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

    def run(self) -> None:
        """"""
        print("MultiplicativeCongruentMethod")

    def __str__(self) -> str:
        """"""
        super().__str__()


class LAB:
    """Construction and research of the characteristics  of sensors."""
    def __init__(self, *method_objects) -> None:
        self.methods: list = list(method_objects)
    
    @property
    def methods(self) -> list:
        """"""
        return self._methods
    
    @methods.setter
    def methods(self, *methods) -> None:
        """"""
        methods = methods[0]
        try:
            for index, method in enumerate(methods):
                if not isinstance(method, BaseMethod):
                    error_text = "{0} not inherited from BaseMethod abstract class".format(method)
                    del methods[index]
                    raise NotInheritedFromBaseMethod(error_text)
                self._methods = methods
        except Exception as ex:
            print(ex)

    
    def start(self) -> None:
        """Function for run all methods"""
        for method in self.methods:
            method.run()


def main() -> None:
    """Main"""
    midsquare_method = MidSquareMethod()
    multiplicative_congruent_method = MultiplicativeCongruentMethod()

    lab = LAB(midsquare_method, multiplicative_congruent_method)
    lab.start()


if __name__ == "__main__":
    """Entry point"""
    main()