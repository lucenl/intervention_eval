from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Treatment(ABC):
    """Base class defining the treatment interface."""
    @abstractmethod
    def apply(self, **kwargs) -> Any:
        pass
