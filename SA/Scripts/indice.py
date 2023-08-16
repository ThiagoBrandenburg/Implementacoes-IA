

from enum import Enum


class Indice(Enum):
    PATHDATABASE = 'pathDatabase'
    GRAFICO = 'grafico'
    NOME = 'NAME'
    COMENTARIO = 'COMMENT'
    TIPO = 'TSP'
    DIMENSOES = 'DIMENSION'
    TIPO_PESO_ARESTA = 'EDGE_WEIGHT_TYPE'
    COORDENADAS = 'NODE_COORD_SECTION'
    EOF = 'EOF'
    ITERACOES = 'iteracoes'
    TEMPERATURA_INICIAL = 'temperatura_inicial'
    TEMPERATURA_FINAL = 'temperatura_final'
    COOLING = 'cooling'

    def __init__(self) -> None:
        pass