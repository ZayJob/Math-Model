"""LAB 2."""

import sys

from abc import ABC, abstractmethod


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


class BaseRandomVariable(ABC):
    """."""
    @abstractmethod
    def _get_rv(self) -> list:
        """get random variable."""
        pass

    @abstractmethod
    def run_statistical_research(self) -> None:
        """run statistical research."""
        pass

    @abstractmethod
    def run_testing_hypotheses(self) -> None:
        """run testing hypotheses."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


class ContinuousRandomVariable(BaseRandomVariable):
    """."""
    def __init__(self, statistical_research: object, hypotheses: object) -> None:
        self.statistical_research: object = statistical_research
        self.hypotheses: object = hypotheses

    @property
    def statistical_research(self) -> object:
        """Get statistical research."""
        return self._statistical_research
    
    @statistical_research.setter
    def statistical_research(self, statistical_research: object) -> None:
        """Set statistical research."""
        self._statistical_research = CheckingInheritance.verify_for_object(statistical_research, BaseStatisticalResearch)

    @property
    def hypotheses(self) -> object:
        """Get hypotheses."""
        return self._hypotheses
    
    @hypotheses.setter
    def hypotheses(self, hypotheses: object) -> None:
        """Set hypotheses."""
        self._hypotheses = CheckingInheritance.verify_for_object(hypotheses, BaseHypotheses)

    def _get_rv(self) -> list:
        """get random variable."""
        pass

    def run_statistical_research(self) -> None:
        """run statistical research."""
        pass

    def run_testing_hypotheses(self) -> None:
        """run testing hypotheses."""
        pass

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class DiscreteRandomVariable(BaseRandomVariable):
    """."""
    def __init__(self, statistical_research: object, hypotheses: object) -> None:
        self.statistical_research: object = statistical_research
        self.hypotheses: object = hypotheses

    @property
    def statistical_research(self) -> object:
        """Get statistical research."""
        return self._statistical_research
    
    @statistical_research.setter
    def statistical_research(self, statistical_research: object) -> None:
        """Set statistical research."""
        self._statistical_research = CheckingInheritance.verify_for_object(statistical_research, BaseStatisticalResearch)

    @property
    def hypotheses(self) -> object:
        """Get hypotheses."""
        return self._hypotheses
    
    @hypotheses.setter
    def hypotheses(self, hypotheses: object) -> None:
        """Set hypotheses."""
        self._hypotheses = CheckingInheritance.verify_for_object(hypotheses, BaseHypotheses)

    def _get_rv(self) -> list:
        """get random variable."""
        pass

    def run_statistical_research(self) -> None:
        """run statistical research."""
        pass

    def run_testing_hypotheses(self) -> None:
        """run testing hypotheses."""
        pass

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class BaseStatisticalResearch(ABC):
    """."""
    @abstractmethod
    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


class CRVStatisticalResearch(BaseStatisticalResearch):
    """."""
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class DRVStatisticalResearch(BaseStatisticalResearch):
    """."""
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class BaseHypotheses(ABC):
    """."""
    @abstractmethod
    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


class CRVHypotheses(BaseHypotheses):
    """."""
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        """__str__."""
        super().__str__()


class DRVHypotheses(BaseHypotheses):
    """."""
    def __init__(self) -> None:
        pass

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
            rv.run_statistical_research()
            rv.run_testing_hypotheses()


class Test:
    def __init__(self):
        pass

    def __str__(self) -> str:
        """__str__."""
        return self.__class__.__name__


def main() -> None:
    """Main"""
    crv_statistical_research = CRVStatisticalResearch()
    drv_statistical_research = DRVStatisticalResearch()

    crv_hypotheses = CRVHypotheses()
    drv_hypotheses = DRVHypotheses()

    continuous_random_variable = ContinuousRandomVariable(
        crv_statistical_research,
        crv_hypotheses,
    )
    discrete_random_variable = DiscreteRandomVariable(
        drv_statistical_research,
        drv_hypotheses
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