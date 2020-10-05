"""LAB 2."""

import sys
import math

import numpy as np

import scipy.integrate as integrate
from matplotlib import pyplot as plt

from random import random
from abc import ABC, abstractmethod


def func(x: float, y: float) -> float:
    """Func"""
    return math.exp(-x-y)


class NotInheritedFromBaseClass(Exception):
    """Ecxeption class"""
    def __init__(self, text):
        self.txt = text


class CheckingInheritance:
    """."""
    @staticmethod
    def verify_for_list_objects(list_object: list, base_class) -> list:
        """Check inheritance."""
        for obj in list_object:
            CheckingInheritance.verify_for_object(obj, base_class)
        return list_object

    @staticmethod
    def verify_for_object(obj: object, base_class) -> object:
        """Check inheritance."""
        try:
            if not isinstance(obj, base_class):
                error_text = "{0} not inherited from {1} abstract class".format(obj, base_class.__name__)
                raise NotInheritedFromBaseClass(error_text)
            return obj
        except Exception as ex:
            print(ex)
            sys.exit()


class BaseStatisticalResearch(ABC):
    """."""
    @abstractmethod
    def M(self, x: list) -> float:
        """Сheckmate waiting M(x)."""
        return 1/len(x) * sum(x)

    @abstractmethod
    def D(self, x: list) -> float:
        """Dispersion D(x)."""
        return 1/len(x) * sum(np.array(x) ** 2 - self.M(x) ** 2)

    @abstractmethod
    def M_x_y(self, x: list, y: list) -> float:
        """Сheckmate waiting M(xy)"""
        return sum([i * j for i, j in zip(x, y)]) * 1/len(x)

    @abstractmethod
    def corrcoef(self, x: list, y: list) -> None:
        """Calculate and print corr coef."""
        return (self.M_x_y(x, y) - self.M(x) * self.M(y)) / math.sqrt(self.D(x) * self.D(y))

    @abstractmethod
    def theoretical_M(self, a: int, b: int, by='x') -> float:
        """Сheckmate waiting M(x)."""
        pass

    @abstractmethod
    def theoretical_D(self, a: int, b: int, by='x') -> float:
        """Dispersion D(x)."""
        pass

    @abstractmethod
    def theoretical_M_x_y(self) -> float:
        """Сheckmate waiting M(xy)"""
        pass

    @abstractmethod
    def theoretical_corrcoef(self, a:int, b: int) -> float:
        """Calculate and print corr coef."""
        pass

    @abstractmethod
    def function_interval(self, func: object, n: int, list_values: list) -> tuple:
        """."""
        result_list = [func(np.random.choice(list_values, n, replace=True)) for _ in range(n)]
        var = np.var(result_list)
        sigma = np.sqrt(var)
        result = func(list_values)
        return (result - 3 * sigma, result + 3 * sigma)

    @abstractmethod
    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


class MixinHist:
    """."""
    def __init__(self):
        pass

    def component_vectors_histogram(self, x: list, y: list) -> None:
        """."""
        x, y = np.meshgrid(sorted(x), sorted(y))
        z = np.exp(-x-y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.set_title("3D hist")
        ax.plot_wireframe(x, y, z)
        plt.show()


class MixinEmpiricalFunction:
    """."""
    def __init__(self):
        pass

    def empirical_function(self, x: list, y: list) -> None:
        """."""
        pass


class CRVStatisticalResearch(BaseStatisticalResearch, MixinHist):
    """."""
    def __init__(self) -> None:
        pass

    def M(self, x: list) -> float:
        """Сheckmate waiting M(x)."""
        return super().M(x)

    def D(self, x: list) -> float:
        """Dispersion D(x)."""
        return super().D(x)

    def M_x_y(self, x: list, y: list) -> float:
        """Сheckmate waiting M(xy)"""
        return super().M_x_y(x, y)

    def corrcoef(self, x: list, y: list) -> None:
        """Calculate and print corr coef."""
        return super().corrcoef(x, y)

    def theoretical_M(self, a: int, b: int, by='x') -> float:
        """Сheckmate waiting theoretical_M(x)."""
        l_f = lambda x, y: func(x, y) * (eval(by))
        l_a = lambda x: a
        l_b = lambda x: b
        return integrate.dblquad(l_f, a, b, l_a, l_b)[0]

    def theoretical_D(self, a: int, b: int, by='x') -> float:
        """Dispersion theoretical_D(x)."""
        l_f = lambda x, y: func(x, y) * (eval(by)) ** 2
        l_a = lambda x: a
        l_b = lambda x: b
        return integrate.dblquad(l_f, a, b, l_a, l_b)[0] - self.theoretical_M(a, b, by) ** 2

    def theoretical_M_x_y(self, a: int, b: int) -> float:
        """Сheckmate waiting theoretical_M_x_y(xy)"""
        return self.theoretical_M(a, b, by='x * y')

    def theoretical_corrcoef(self, a:int, b: int) -> float:
        """Calculate and print corr coef."""
        return (self.theoretical_M_x_y(a, b) - self.theoretical_M(a, b, by='x') * self.theoretical_M(a, b, by='y')) / math.sqrt(self.theoretical_D(a, b, by='x') * self.theoretical_D(a, b, by='y'))

    def function_interval(self, func: object, n: int, list_values: list) -> tuple:
        """."""
        return super().function_interval(func, n, list_values)

    def component_vectors_histogram(self, x: list, y: list) -> None:
        """."""
        super().component_vectors_histogram(x, y)

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class DRVStatisticalResearch(BaseStatisticalResearch, MixinEmpiricalFunction, MixinHist):
    """."""
    def __init__(self) -> None:
        pass

    def M(self, x: list) -> float:
        """Сheckmate waiting M(x)."""
        return super().M(x)

    def D(self, x: list) -> float:
        """Dispersion D(x)."""
        return super().D(x)

    def M_x_y(self, x: list, y: list) -> float:
        """Сheckmate waiting M(xy)"""
        return super().M_x_y(x, y)

    def corrcoef(self, x: list, y: list) -> None:
        """Calculate and print corr coef."""
        return super().corrcoef(x, y)

    def theoretical_M(self, a: int, b: int, by='x') -> float:
        """Сheckmate waiting theoretical_M(x)."""
        l_f = lambda x, y: func(x, y) * (eval(by))
        l_a = lambda x: a
        l_b = lambda x: b
        return integrate.dblquad(l_f, a, b, l_a, l_b)[0]

    def theoretical_D(self, a: int, b: int, by='x') -> float:
        """Dispersion theoretical_D(x)."""
        l_f = lambda x, y: func(x, y) * (eval(by)) ** 2
        l_a = lambda x: a
        l_b = lambda x: b
        return integrate.dblquad(l_f, a, b, l_a, l_b)[0] - self.theoretical_M(a, b, by) ** 2

    def theoretical_M_x_y(self, a: int, b: int) -> float:
        """Сheckmate waiting theoretical_M_x_y(xy)"""
        return self.theoretical_M(a, b, by='x * y')

    def theoretical_corrcoef(self, a:int, b: int) -> float:
        """Calculate and print corr coef."""
        return (self.theoretical_M_x_y(a, b) - self.theoretical_M(a, b, by='x') * self.theoretical_M(a, b, by='y')) / math.sqrt(self.theoretical_D(a, b, by='x') * self.theoretical_D(a, b, by='y'))

    def function_interval(self, func: object, n: int, list_values: list) -> tuple:
        """."""
        return super().function_interval(func, n, list_values)

    def component_vectors_histogram(self, x: list, y: list) -> None:
        """."""
        super().component_vectors_histogram(x, y)

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class BaseRandomVariable(ABC):
    """."""
    @abstractmethod
    def neumanns_method(self) -> list:
        """Neumann's method."""
        pairs = []
        i = 0
        while i < self.n:
            pair = [random(), random()]
            new_max_value = random() * self.max_value
            if func(*pair) >= new_max_value:
                pairs.append(pair)
                i += 1
        return pairs

    @abstractmethod
    def run(self) -> None:
        """."""
        pass

    @abstractmethod
    def run_statistical_research(self) -> None:
        """run statistical research."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


class ContinuousRandomVariable(BaseRandomVariable):
    """."""
    def __init__(self,n: int, func: object, statistical_research: object) -> None:
        self.statistical_research: object = statistical_research
        self.func: object = func
        self.a: int = 0
        self.b: int = 1
        self.n: int = n
        self.max_value = 1
        self.pairs: list = []

    @property
    def statistical_research(self) -> object:
        """Get statistical research."""
        return self._statistical_research
    
    @statistical_research.setter
    def statistical_research(self, statistical_research: object) -> None:
        """Set statistical research."""
        self._statistical_research = CheckingInheritance.verify_for_object(statistical_research, BaseStatisticalResearch)

    def neumanns_method(self) -> list:
        """Neumann's method."""
        self.pairs = super().neumanns_method()

    def run(self) -> None:
        """."""
        self.neumanns_method()

    def run_statistical_research(self) -> None:
        """run statistical research."""
        x, y = list(zip(*self.pairs))

        self.statistical_research.component_vectors_histogram(x, y)

        print(f"M(x) = {self.statistical_research.M(x)}")
        print(f"Interval M(x) = {self.statistical_research.function_interval(self.statistical_research.M, 1000, x)}")
        print(f"Theoretical M(x) = {self.statistical_research.theoretical_M(self.a, self.b, by='x')}\n")
    
        print(f"D(x) = {self.statistical_research.D(x)}")
        print(f"Interval D(x) = {self.statistical_research.function_interval(self.statistical_research.D, 1000, x)}")
        print(f"Theoretical D(x) = {self.statistical_research.theoretical_D(self.a, self.b, by='x')}\n")

        print(f"M(y) = {self.statistical_research.M(y)}")
        print(f"Interval M(y) = {self.statistical_research.function_interval(self.statistical_research.M, 1000, y)}")
        print(f"Theoretical M(y) = {self.statistical_research.theoretical_M(self.a, self.b, by='y')}\n")

        print(f"D(y) = {self.statistical_research.D(y)}")
        print(f"Interval D(y) = {self.statistical_research.function_interval(self.statistical_research.D, 1000, x)}")
        print(f"Theoretical D(y) = {self.statistical_research.theoretical_D(self.a, self.b, by='y')}\n")

        print(f"Corrcoef = {self.statistical_research.corrcoef(x, y)}")
        print(f"Theoretical Corrcoef = {self.statistical_research.theoretical_corrcoef(self.a, self.b)}\n")

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class DiscreteRandomVariable(BaseRandomVariable):
    """."""
    def __init__(self,n: int, func: object, statistical_research: object) -> None:
        self.statistical_research: object = statistical_research
        self.func: object = func
        self.a: int = 0
        self.b: int = 1
        self.n: int = n
        self.max_value = 1
        self.pairs: list = []

    @property
    def statistical_research(self) -> object:
        """Get statistical research."""
        return self._statistical_research
    
    @statistical_research.setter
    def statistical_research(self, statistical_research: object) -> None:
        """Set statistical research."""
        self._statistical_research = CheckingInheritance.verify_for_object(statistical_research, BaseStatisticalResearch)

    def neumanns_method(self) -> list:
        """Neumann's method."""
        self.pairs = super().neumanns_method()

    def run(self) -> None:
        """."""
        self.neumanns_method()

    def run_statistical_research(self) -> None:
        """run statistical research."""
        x, y = list(zip(*self.pairs))

        self.statistical_research.empirical_function(x, y)

        print(f"M(x) = {self.statistical_research.M(x)}")
        print(f"Interval M(x) = {self.statistical_research.function_interval(self.statistical_research.M, 1000, x)}")
        print(f"Theoretical M(x) = {self.statistical_research.theoretical_M(self.a, self.b, by='x')}\n")
    
        print(f"D(x) = {self.statistical_research.D(x)}")
        print(f"Interval D(x) = {self.statistical_research.function_interval(self.statistical_research.D, 1000, x)}")
        print(f"Theoretical D(x) = {self.statistical_research.theoretical_D(self.a, self.b, by='x')}\n")

        print(f"M(y) = {self.statistical_research.M(y)}")
        print(f"Interval M(y) = {self.statistical_research.function_interval(self.statistical_research.M, 1000, y)}")
        print(f"Theoretical M(y) = {self.statistical_research.theoretical_M(self.a, self.b, by='y')}\n")

        print(f"D(y) = {self.statistical_research.D(y)}")
        print(f"Interval D(y) = {self.statistical_research.function_interval(self.statistical_research.D, 1000, x)}")
        print(f"Theoretical D(y) = {self.statistical_research.theoretical_D(self.a, self.b, by='y')}\n")

        print(f"Corrcoef = {self.statistical_research.corrcoef(x, y)}")
        print(f"Theoretical Corrcoef = {self.statistical_research.theoretical_corrcoef(self.a, self.b)}\n")

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class LAB:
    """."""
    def __init__(self) -> None:
        self.random_variables: list = []

    def create_list_of_random_variables(self, *random_variable_objects: tuple) -> None:
        """Create list of methods."""
        self.random_variables: list = list(random_variable_objects)

    @property
    def random_variables(self) -> list:
        """Get methods."""
        return self._random_variables
    
    @random_variables.setter
    def random_variables(self, *random_variables: tuple) -> None:
        """Set random values."""
        random_variables = random_variables[0]
        self._random_variables = CheckingInheritance.verify_for_list_objects(random_variables, BaseRandomVariable)

    def run(self) -> None:
        """."""
        for rv in self.random_variables:
            rv.run()
            rv.run_statistical_research()


def main() -> None:
    """Main"""
    crv_statistical_research = CRVStatisticalResearch()
    drv_statistical_research = DRVStatisticalResearch()

    continuous_random_variable = ContinuousRandomVariable(
        1000,
        func,
        crv_statistical_research
    )

    discrete_random_variable = DiscreteRandomVariable(
        1000,
        func,
        drv_statistical_research
    )

    lab = LAB()
    lab.create_list_of_random_variables(
        continuous_random_variable,
        discrete_random_variable
    )
    lab.run()


if __name__ == "__main__":
    """Entry point"""
    main()