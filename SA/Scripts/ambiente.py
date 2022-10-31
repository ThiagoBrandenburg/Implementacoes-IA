import imp
from math import sqrt
from indice import Indice

indice = Indice

class Ponto:
    
    posicao : tuple[float, float]
    
    def __init__(self, x: float, y: float) -> None:
        self.posicao = (x,y)
    
    def __init__(self, posicao: tuple[float, float]) -> None:
        self.posicao = posicao
    

class Aresta:
    origem: Ponto
    destino: Ponto
    peso : float

    def __init__(self, origem: Ponto, destino: Ponto, peso: float) -> None:
        self.origem = origem
        self.destino = destino
        self.peso = peso
        

class Grafo:
    pontos : dict[Ponto, list[Aresta]]

    def __init__(self) -> None:
        pass

    def distanciaEuclidiana(p1: Ponto, p2: Ponto)-> float:
        return sqrt(pow(p1.posicao[0] -p2.posicao[0], 2) + pow(p1.posicao[1] - p2.posicao[1], 2))


        