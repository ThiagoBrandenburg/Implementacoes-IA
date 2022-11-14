'''

from ambiente import Ambiente
import pygame

class Exibicao:
    ambiente: Ambiente
    PRETO = (0,0,0)
    BRANCO = (255,255,255)
    VERMELHO = (255,0,0)
    CINZA_ESCURO = (50,50,50)
    CINZA_CLARO = (200,200,200)
    AZUL_ESCURO = (0,0,100)
    #COR_TAGS = {1: (0,0,250), 2:(0,0,200), 3:(0,0,150), 4:(0,0,100)}

    def __init__(self,
                ambiente :  Ambiente,
                escala = 0.5,
                corFundo = CINZA_ESCURO,
                corPonto = ) -> None:
        #Declarando parametros
        self.dimensaoXCelula = resolucaoCelula[0]
        self.dimensaoYCelula = resolucaoCelula[1]
        self.ambiente = ambiente

        #cores
        self.corFundo = corFundo
        self.corCelula = corCelula
        self.corAgente = corAgente
        self.corItem = corItem

        #tela iniciada, talvez passar para uma funcao start depois
        self.resolucaoTela = (self.ambiente.nLinhas*self.dimensaoYCelula, self.ambiente.nColunas*self.dimensaoXCelula)
        self.tela = pygame.display.set_mode(self.resolucaoTela)
        pygame.font.init()
        self.fonte = pygame.font.SysFont('Arial',16)
        pygame.display.get_surface().fill(corFundo)

    def exibeRetangulo(self, x, y, cor):
        pygame.draw.rect(self.tela, cor, [x, y, self.dimensaoXCelula, self.dimensaoYCelula])

    def exibeNumero(self, numero: int, x, y):
        #imagem_texto = pygame.font.Font.render()
        imagem_texto = self.fonte.render(str(numero), True, (255,255,255))
        self.tela.blit(imagem_texto, (x,y))


    def exibeCelula(self, x, y):
        #gera Celula
        posicaoX = x*self.dimensaoXCelula
        posicaoY = y*self.dimensaoYCelula

        #posicao da celula
        celula = self.ambiente.mapa[x][y]
        cor = self.corCelula
        if celula.temItem():
            cor = self.corItem
            pygame.draw.rect(self.tela, cor, [posicaoX+1, posicaoY+1, self.dimensaoXCelula -2, self.dimensaoYCelula -2])
            self.exibeNumero(celula.item.tag, posicaoX, posicaoY)
        else:
            pygame.draw.rect(self.tela, cor, [posicaoX+1, posicaoY+1, self.dimensaoXCelula -2, self.dimensaoYCelula -2])
            

    def exibeAgente(self, agente: Agente):
        posicaoX = agente.posicao.x * self.dimensaoXCelula
        posicaoY = agente.posicao.y * self.dimensaoYCelula
        pygame.draw.rect(self.tela, self.corAgente, [posicaoX +self.dimensaoXCelula/3, posicaoY +self.dimensaoYCelula/3, self.dimensaoXCelula/2, self.dimensaoYCelula/2])
        if agente.carregaItem():
            pygame.draw.rect(self.tela, self.corItem, [posicaoX + self.dimensaoXCelula/3, posicaoY + self.dimensaoYCelula/3, self.dimensaoXCelula/3, self.dimensaoYCelula/3])


    def exibeFrame(self):
        self.tela.fill(self.corFundo)
        for x in range(self.ambiente.nLinhas):
            for y in range(self.ambiente.nColunas):
                self.exibeCelula(x,y)

        for agente in self.ambiente.listaAgente:
            self.exibeAgente(agente)

        pygame.display.update()
'''
from ambiente import Ambiente
import matplotlib.pyplot as plt

class Exibicao:
    ambiente : Ambiente


    def __init__(self, ambiente: Ambiente):
        self.ambiente = ambiente
    

    def exibirFrame(self):
        l = self.ambienteParaPyplot()
        fecho = self.fecho()
        
        plt.plot(l[0],l[1],color='blue')
        plt.plot(fecho[0],fecho[1],color='blue')
        plt.plot(l[0],l[1],'o',color='black')
        plt.show()

