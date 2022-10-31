from ambiente import Ambiente
from exibicao import Exibicao
from leitor import Leitor
import pygame



class Controle:
    leitor: Leitor
    ambiente: Ambiente
    exibicao: Exibicao
    configuracoes: dict

    PATH_DADOS = 'dados'
    ITERACOES = 'iteracoes'
    DIMENSAOX = 'dimensaoX'
    DIMENSAOY = 'dimensaoY'
    NUMERO_DE_AGENTES  = 'numero_de_agentes'
    VISAO = 'visao'
    ALPHA = 'alpha'

    def __init__(self) -> None:
        self.leitor = None
        self.configuracoes = None
        self.ambiente = None
        self.exibicao = None


    def carregar(self):
        self.leitor = Leitor()
        self.configuracoes = self.leitor.carregaConfiguracoes()
        self.ambiente = Ambiente(self.configuracoes)
        self.exibicao = Exibicao(self.ambiente)

    def startR15Database(self):
        dados = self.leitor.carregaDatabaseR15(self.configuracoes[self.PATH_DADOS])
        self.ambiente.carregaDados(dados)
    
    def run(self):
        self.carregar()
        print('(Controle) Configuracoes Carregadas')
        self.startR15Database()
        print('(Controle) Dados Carregados')
        self.ambiente.inicia()
        print('(Controle) Ambiente Inicializado')
        contador = 0
        for i in range(self.configuracoes[self.ITERACOES]):
            contador +=1
            self.ambiente.iteracao()
            if contador > 1000:
                self.exibicao.exibeFrame()
                contador = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        print('(Controle) Numero de Iteracoes Alcancado, entrando em modo de finalizacao')
        while not self.ambiente.iteracaoFinal():
            #self.exibicao.exibeFrame()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        print('(Controle) Execucao Finalizada')
        while True:
            self.exibicao.exibeFrame()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()





