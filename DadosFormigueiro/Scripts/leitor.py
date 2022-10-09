from code import interact
from typing import Dict
from ambiente import Item
import json

class Leitor:
    pathConfig: str
    pathDatabase: str

    def __init__(self) -> None:
        pass
    
    
    def carregaConfiguracoes(self, pathConfig = 'DadosFormigueiro/config.txt')-> Dict:
        d = {}
        try:
            with open(pathConfig) as arquivo:
                dados = arquivo.read()
                d = json.loads(dados)
            return d
        except: Exception('Erro de leitura das configuracoes')

    def carregaDatabaseR15(self, pathDatabase: str)->list[Item]:
        itens = []
        with open(pathDatabase) as arquivo:
            for linha in arquivo.readlines():
                if not linha[0] == '#':
                    try:
                        l = linha.split()
                        t = (l[0],l[1],l[2])
                        itens.append(Item(t))
                    except: 'linha vazia'
        return itens



