import numpy as np
from math import sqrt

# sterge


class Rectangle:
    def __init__(self, rect, weight:int = -1) -> None:
        self.rect = rect
        self.weight:int = weight 
    
    def __repr__(self) -> str:
        return f"([{self.rect[0]}, {self.rect[1]}, {self.rect[2]}, {self.rect[3]}], {self.weight})"



