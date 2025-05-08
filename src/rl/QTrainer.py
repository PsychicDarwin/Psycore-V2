from abc import ABC, abstractmethod
# This class takes the Q Loader with it's attributes, and performs the Q Learning algorithm for training
class QTrainer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def cost_function(self):
        pass