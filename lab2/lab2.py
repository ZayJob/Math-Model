"""LAB 2."""

import math
import sys

from abc import ABC, abstractmethod
from collections import Counter
from pprint import pprint as pp
from random import random

import numpy as np
import scipy.integrate as integrate

from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


X_ARRAY = [v for v in range(1, 11, 2)]

Y_ARRAY = [v for v in range(1, 16, 3)]

Pxy_ARRAY = [
    [0.05, 0.025, 0.075, 0.01, 0.04],
    [0.07, 0.01, 0.02, 0.055, 0.045],
    [0.025, 0.035, 0.05, 0.06, 0.03],
    [0.10, 0.01, 0.04, 0.025, 0.025],
    [0.05, 0.025, 0.02, 0.08, 0.025]
]


def func(x: float, y: float) -> float:
    """Func"""
    return math.exp(-x-y)


class NotInheritedFromBaseClass(Exception):
    """Ecxeption class"""
    def __init__(self, text):
        self.txt = text


class CheckingInheritance:
    """CheckingInheritance class."""
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
    """BaseStatisticalResearch class."""
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
        """Search function interval."""
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
    """MixinHist class."""
    def __init__(self):
        pass

    def component_vectors_histogram(self, x: list, y: list) -> None:
        """Create and show component vectors histogram."""
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
    """MixinEmpiricalFunction class."""
    def __init__(self):
        pass

    def empirical_matrix_histogram(self, x: list, y: list, n: int) -> None:
        """Create and show empirical matrix 3d histogram."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        hist, xedges, yedges = np.histogram2d(x, y, bins=(len(Counter(x)), len(Counter(y))), range=[[0, max(x)], [0, max(y)]])

        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = dy = np.ones_like(zpos)
        dz = hist.ravel() / n

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.set_title("Empirical matrix 3D hist")
        plt.show()


class CRVStatisticalResearch(BaseStatisticalResearch, MixinHist):
    """CRVStatisticalResearch class."""
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
        """Search function interval."""
        return super().function_interval(func, n, list_values)

    def component_vectors_histogram(self, x: list, y: list) -> None:
        """Create and show component vectors histogram."""
        super().component_vectors_histogram(x, y)

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class DRVStatisticalResearch(BaseStatisticalResearch, MixinEmpiricalFunction):
    """DRVStatisticalResearch class."""
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
        """Search function interval."""
        return super().function_interval(func, n, list_values)

    def empirical_matrix_histogram(self, x: list, y: list, n: int) -> None:
        """Create and show empirical matrix 3d histogram."""
        super().empirical_matrix_histogram(x, y, n)

    def empirical_matrix(self, x: list, y: list, n: int) -> None:
        """Create and print emperical matrix."""
        collection_of_same_pair = Counter(zip(x, y))
        empirical_matrix = np.zeros((5, 5))

        for pair, v in collection_of_same_pair.items():
            empirical_matrix[X_ARRAY.index(pair[0])][Y_ARRAY.index(pair[1])] = v / n
        
        print(f"Emperical matrix:")
        pp(list(map(list, empirical_matrix)))
        print("\n")

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class BaseRandomVariable(ABC):
    """BaseRandomVariable class."""
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
        """Run RV."""
        pass

    @abstractmethod
    def run_statistical_research(self) -> None:
        """Run statistical research."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


class ContinuousRandomVariable(BaseRandomVariable):
    """ContinuousRandomVariable class."""
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
        """Run CRV."""
        print(f"---------RUN CRV---------\n")
        self.neumanns_method()

    def run_statistical_research(self) -> None:
        """Run statistical research."""
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
        print(f"---------END CRV---------\n")

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class DiscreteRandomVariable(BaseRandomVariable):
    """DiscreteRandomVariable class."""
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

    def get_index_biggest_value(self, list_values: list, num: float) -> int:
        """Get index biggest the num value."""
        for v in list_values:
            if v >= num:
                return list_values.index(v) - 1

    def run(self) -> None:
        """Run DRV."""
        print(f"---------RUN DRV---------\n")
        self.neumanns_method()

        x_p = [sum(_) for _ in Pxy_ARRAY]
        print(f"X probabilities = {x_p}")

        func_x_p = [round(sum(x_p[:i]), 6) for i, v in enumerate(x_p)] + [1.0]
        print(f"X probabilities function = {func_x_p}\n")

        y_matrix = [list(map(lambda x: round(x / x_p[i], 6), Pxy_ARRAY[i])) for i, line in enumerate(Pxy_ARRAY)]
        print(f"Y matrix:")
        pp(y_matrix)
        print("\n")

        func_y_p = [[round(sum(row[:i]), 6) for i, v in enumerate(row)] + [1.0] for row in y_matrix]
        print(f"Y probabilities function = ")
        pp(func_y_p)
        print("\n")

        self.pairs = [[X_ARRAY[self.get_index_biggest_value(func_x_p, x)], y] for x, y in self.pairs]
        self.pairs = [[x, Y_ARRAY[self.get_index_biggest_value(func_y_p[X_ARRAY.index(x)], y)]] for x, y in self.pairs]

    def run_statistical_research(self) -> None:
        """Run statistical research."""
        x, y = list(zip(*self.pairs))

        self.statistical_research.empirical_matrix(x, y, self.n)

        self.statistical_research.empirical_matrix_histogram(x, y, self.n)

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
        print(f"---------END DRV---------\n")

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class LAB:
    """LAB class."""
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
        """Run LAB1"""
        for rv in self.random_variables:
            rv.run()
            rv.run_statistical_research()


def main() -> None:
    """Main."""
    crv_statistical_research = CRVStatisticalResearch()
    drv_statistical_research = DRVStatisticalResearch()

    continuous_random_variable = ContinuousRandomVariable(
        10000,
        func,
        crv_statistical_research
    )

    discrete_random_variable = DiscreteRandomVariable(
        10000,
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
    """Entry point."""
    main()