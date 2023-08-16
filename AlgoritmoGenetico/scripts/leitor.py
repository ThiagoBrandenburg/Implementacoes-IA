from pathlib import Path
from time import sleep
from typing import Dict
import json


class Leitor:
    pathConfig: str
    pathDatabase: str

    def __init__(self) -> None:
        pass

    
    def mapConfig(x:str)->tuple[str,str]:
        key,value = map(str.strip,x.split('='))
        return key,value


    def loadConfig(self, pathDatabase: str) -> dict:
        # path = Path(pathDatabase)
        # print(path.absolute())
        try:
            #print('Bom dia')
            with open(pathDatabase) as arquivo:
                linhas = arquivo.readlines()
            print(linhas)
            config = {elemento[0]:elemento[1] for elemento in map(Leitor.mapConfig,linhas)}
            return config 
        except: Exception('(leitor) Erro de leitura de arquivo')

    
    def carregaConfiguracoes(self, pathConfig = 'SA/config.txt')-> Dict:
        d = {}
        try:
            with open(pathConfig) as arquivo:
                dados = arquivo.read()
                d = json.loads(dados)
            return d   
        except: Exception('Erro de leitura das configuracoes')


