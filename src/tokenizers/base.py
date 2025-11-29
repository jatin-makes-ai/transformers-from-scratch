from abc import ABC, abstractmethod

class Tokenizer(ABC):

    @abstractmethod
    def encode(self, text: str) -> list:
        pass

    @abstractmethod
    def decode(self, ids: list) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
