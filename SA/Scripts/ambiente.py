import math
from random import randrange,random
from indice import Indice
import matplotlib.pyplot as plt


class Ambiente:
    dimensao: int
    pontos : list[tuple[float,float]]
    matriz : list[list[float]]
    caminhoAtual : list[int]
    iteracoes : int
    t_inicial : float
    t_final : float
    t_atual : float
    iteracoes : int
    i_atual : int
    pesocaminhoAtual : float
    chance : float
    historico: list[list]

    def __init__(self, base_de_dados : dict, configuracoes: dict):
        self.dimensao = int(base_de_dados[Indice.DIMENSOES][0])
        self.pontos = [elemento for elemento in base_de_dados[Indice.COORDENADAS]]
        self.matriz = [[0.0 for i in range(self.dimensao)] for j in range(self.dimensao)]
        self.caminhoAtual = []
        self.pesocaminhoAtual = 0

        self.t_inicial = float(configuracoes[Indice.TEMPERATURA_INICIAL])
        self.t_final = float(configuracoes[Indice.TEMPERATURA_FINAL])
        self.t_atual = self.t_inicial
        self.iteracoes = int(configuracoes[Indice.ITERACOES])
        self.i_atual = 0
        self.chance = 0
        self.historico = [[],[],[]]

        self.cooling = int(configuracoes[Indice.COOLING])
        self.cooling_func = self.cooling0


    def distanciaEuclidiana(self, t1: tuple[float,float], t2: tuple[float,float])->float:
        return math.sqrt(pow(t1[0]-t2[0], 2) + pow(t1[1]-t2[1], 2))

    def iniciaMatriz(self):
        for i in range(self.dimensao):
            for j in range(i):
                distancia = self.distanciaEuclidiana(self.pontos[i], self.pontos[j])
                self.matriz[i][j] = distancia
                self.matriz[j][i] = distancia
    
    def setCaminho(self, caminho: list[int], peso_caminho: float):
        self.caminhoAtual = caminho
        self.pesocaminhoAtual = peso_caminho

    def pesaCaminho(self, l : list[int])->float:
        custo = 0
        tamanho = len(l)
        for i in range(tamanho-1):
            custo += self.matriz[l[i]][l[i+1]]
        custo += self.matriz[self.caminhoAtual[tamanho-1]][0]
        return custo

    def iniciacaminhoAtual(self):
        self.caminhoAtual = []
        opcoes = [i for i in range(self.dimensao)]
        while len(opcoes) > 0:
            escolha = randrange(len(opcoes))
            self.caminhoAtual.append(opcoes.pop(escolha))
        inicio = self.caminhoAtual[0]
        self.caminhoAtual.append(inicio)
        self.pesocaminhoAtual = self.pesaCaminho(self.caminhoAtual)
        

    def limitantes(self):
        x = 0.0
        y = 0.0
        for ponto in self.pontos:
            if ponto[0] > x: x = ponto[0]
            if ponto[1] > y: y = ponto[1]
        return (x,y)

    def pesoAresta(self, pontos  = tuple[int,int])->float:
        return self.matriz[pontos[0]][pontos[1]]

    def pesoAresta(self, p1 = int, p2 = int)->float:
        return self.matriz[p1][p2]
    

    def pertubacao(self, n: int)->list[int]:
        '''Causa uma pertubacao no caminhoAtual'''
        l = self.caminhoAtual.copy()
        for i in range(n):
            a = randrange(1,len(l)-1)
            b = randrange(1,len(l)-1)
            while a==b:
                b = randrange(1,len(l)-1)
            aux = l[a]
            l[a] = l[b]
            l[b] = aux
        return l
        
    def cooling0(self):
        self.t_atual = self.t_inicial - self.i_atual*((self.t_inicial - self.t_final)/self.iteracoes)

    def cooling1(self):
        self.t_atual = self.t_inicial*pow(self.t_final/self.t_inicial, self.i_atual/self.iteracoes)

    def cooling3(self):
        a = (math.log(self.t_inicial - self.t_final))/(math.log(self.iteracoes))
        self.t_atual = self.t_inicial - pow(self.i_atual, a)
    
    def cooling4(self):
        self.t_atual = ((self.t_inicial - self.t_final)/(1 + math.pow(math.e, 0.3*(self.i_atual - self.iteracoes/2)))) + self.t_final
    
    def cooling5(self):
        self.t_atual = (0.5*(self.t_inicial - self.t_final)) * (1 + math.cos((self.i_atual*math.pi)/self.iteracoes)) * (self.t_final)


    def set_cooling(self):
        if self.cooling == 0:
            self.cooling_func = self.cooling0
        elif self.cooling ==1:
            self.cooling_func = self.cooling1
        elif self.cooling == 3:
            self.cooling_func = self.cooling3
        elif self.cooling == 4:
            self.cooling_func = self.cooling4
        elif self.cooling == 5:
            self.cooling_func = self.cooling5

    


    def iteracao(self)->bool:
        self.historico[0].append(self.i_atual)
        self.historico[1].append(self.pesocaminhoAtual)
        self.historico[2].append(self.t_atual)
        self.cooling_func()
        if self.t_atual <= self.t_final or self.i_atual >= self.iteracoes:
            print('peso:',self.pesocaminhoAtual, 't_atual',self.t_atual, 't_final',self.t_final, 'i_atual', self.i_atual, 'iteracoes',self.iteracoes)
            return False
        else:
            novo_caminho = self.pertubacao(randrange(5))
            peso_novo_caminho = self.pesaCaminho(novo_caminho)
            delta = peso_novo_caminho - self.pesocaminhoAtual
            if delta < 0:
                self.setCaminho(novo_caminho,peso_novo_caminho)
            else:
                self.chance = math.pow(math.e, (delta*-1)/self.t_atual)
                if random() <= self.chance: self.setCaminho(novo_caminho,peso_novo_caminho)
            self.i_atual += 1
        return True
    
    def exibir(self):
        fig, (ax1, ax2) = plt.subplots(2,1)
        fig.subplots_adjust(hspace=0.5)

        ax1.plot(self.historico[0],self.historico[1])
        ax1.set_xlabel('Iteração')
        ax1.set_ylabel('Peso do caminho')
        ax1.grid(True)
        texto = '%0.2f' % self.historico[1][len(self.historico[1])-1]
        ax1.annotate(texto,xy=(self.historico[0][len(self.historico[0])-1], self.historico[1][len(self.historico[1])-1]), xytext = (0,0), textcoords='offset points')

        ax2.plot(self.historico[0],self.historico[2])
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('Temperatura')
        ax2.grid(True)


        fig.legend('Gráficos de desempenho e temperatura')
        

        plt.show()
    