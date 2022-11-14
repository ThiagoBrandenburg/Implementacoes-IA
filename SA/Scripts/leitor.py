from time import sleep
from indice import Indice
from typing import Dict
import json


class Leitor:
    pathConfig: str
    pathDatabase: str

    def __init__(self) -> None:
        pass

    def carregaDatabase(self, pathDatabase: str) -> dict:
        d = {}
        pontos = []
        try:
            #print('Bom dia')
            with open(pathDatabase) as arquivo:
                linha = arquivo.readline()
                l= linha.replace(':','')
                #print(l)
                elementos = l.split()
                cabecalho = elementos.pop(0)
                d[cabecalho] = elementos
                while cabecalho != Indice.COORDENADAS:
                    #sleep(0.1)
                    linha = arquivo.readline()
                    l = linha.replace(':','')
                    #print(l)
                    elementos = l.split()
                    cabecalho = elementos.pop(0)
                    d[cabecalho] = elementos
                    
                linha = arquivo.readline()
                while linha != 'EOF':
                    #sleep(0.1)
                    elementos = linha.split()
                    print(elementos)
                    coordenada = (float(elementos[1]), float(elementos[2]))
                    pontos.append(coordenada)
                    d[Indice.COORDENADAS] = pontos
                    linha = arquivo.readline()
                    
        except: Exception('(leitor) Erro de leitura de arquivo')

        return d  
    
    def carregaConfiguracoes(self, pathConfig = 'SA/config.txt')-> Dict:
        d = {}
        try:
            with open(pathConfig) as arquivo:
                dados = arquivo.read()
                d = json.loads(dados)
            return d   
        except: Exception('Erro de leitura das configuracoes')


