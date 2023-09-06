from math import sqrt
from random import randrange, random



class Item:
    tag: int
    x: float
    y: float
    
    def __init__(self, tupla: tuple) -> None:
        self.x = float(tupla[0])
        self.y = float(tupla[1])
        self.tag = int(tupla[2])
    
    def tupla(self)-> tuple[float,float,int]:
        return (self.x, self.y, self.tag)

class Celula:
    temAgente: bool
    item: Item
    x: int
    y: int
    def __init__(self,
                posicao: tuple[int,int],
                temAgente = False) -> None:
        self.item = None
        self.temAgente = temAgente
        self.x = posicao[0]
        self.y = posicao[1]
    
    def tupla(self)->tuple:
        return (self.x, self.y)
    
    def temAgente(self):
        return self.temAgente

    def getItem(self)->Item:
        if self.temItem():
            item = self.item
            self.item = None
            return item
        else: raise Exception('Nao ha item para ser retirado')
    
    def setItem(self, item: Item):
        if not self.temItem():
            self.item = item
        else: raise Exception('Ja ha um item nessa celula')

    def temItem(self):
        return self.item is not None



class Agente:
    visao: int
    posicao: Celula
    expoente: int
    item: Item
    def __init__(self,
                limX: int,
                limY: int,
                visao=1,
                expoente=2,
                limitante = 0.01) -> None:
        self.visao = visao
        self.posicao = None

        self.limX = limX
        self.limY = limY
    
        self.expoente = expoente
        self.limitante = limitante

        self.item = None

    def moveParaCelula(self, celula: Celula):
        self.posicao.temAgente = False
        self.posicao = celula
        self.posicao.temAgente = True

    def iniciaAgente(self, celula: Celula):
        self.posicao = celula
        self.posicao.temAgente = True
        
    def carregaItem(self):
        return self.item is not None

    def temItem(self):
        return self.item is not None 

    def getItem(self)->Item:
        if self.temItem():
            item = self.item
            self.item = None
            return item
        else: raise Exception('Nao ha item para ser retirado')
    
    def setItem(self, item: Item):
        if not self.temItem():
            self.item = item
        else: raise Exception('Ja ha um item nessa celula')

    def passo(self, vizinhos: list[Celula]):
        vizinhos_disponiveis = []
        for vizinho in vizinhos:
            if vizinho.temAgente is False:
                vizinhos_disponiveis.append(vizinho)
        if len(vizinhos_disponiveis)>0:
            escolha = randrange(0,len(vizinhos_disponiveis))
            self.moveParaCelula(vizinhos_disponiveis[escolha])

    def pegar(self):
        if self.posicao.temItem() and (not self.carregaItem()):
            self.setItem(self.posicao.getItem())
        else: raise Exception('Erro ao pegar')
    
    def largar(self):
        if (not self.posicao.temItem()) and self.carregaItem():
            self.posicao.setItem(self.getItem())
        else: raise Exception('Erro ao largar')

    def fator(self, vizinhos: list[Celula]):
        itens = 0
        for vizinho in vizinhos:
            if vizinho.temItem():
                itens +=1
        return itens/len(vizinhos)

    def tomadaDecisaoIngenua(self, vizinhos: list[Celula]):
        chance = pow(self.fator(vizinhos),self.expoente)# se tem muitos, decisao a sim
        decisao = random() < chance
        if self.carregaItem() and (not self.posicao.temItem()) and decisao:
                self.largar()
        if (not self.carregaItem()) and (self.posicao.temItem()) and (not decisao):
                self.pegar()     
        self.passo(vizinhos)
    
    def tomadaDecisaoIngenuaComVisao(self, vizinhos: list[Celula], visao: list[Celula]):
        chance = pow(self.fator(visao),self.expoente)
        # se tem muitos, decisao a sim
        decisao = random() < chance
        if self.carregaItem() and (not self.posicao.temItem()) and decisao:
                self.largar()
        if (not self.carregaItem()) and (self.posicao.temItem()) and (not decisao):
                self.pegar()
        
        self.passo(vizinhos)
 
    def tomadaDecisaoIngenuaFinal(self, vizinhos: list[Celula], visao: list[Celula]):
        chance = pow(self.fator(visao),self.expoente)
        decisao = random() < chance
        if self.carregaItem() and (not self.posicao.temItem()) and decisao:
                self.largar()
        self.passo(vizinhos)
    

    def tomadaDecisaoComFatoresK(self, vizinhos: list[Celula], visao: list[Celula], k1: int, k2: int):
        if (not self.carregaItem()) and self.posicao.temItem():
            chance = self.fator(visao)
            chancePegar = pow(k1/ (k1 + chance), self.expoente)
            if (chancePegar > self.limitante) and (random() < chancePegar): self.pegar()
        elif self.carregaItem() and (not self.posicao.temItem()):
            chance = self.fator(vizinhos)
            chanceLargar = pow(chance/ (k2 + chance), 2)
            if (chanceLargar > self.limitante) and (random() < chanceLargar): self.largar()

        self.passo(vizinhos)
    
    def tomadaDecisaoComFatoresFinal(self, vizinhos:list[Celula], visao: list[Celula], k2: int):
        if self.carregaItem() and (not self.posicao.temItem()):
            chance = self.fator(visao)
            chanceLargar = pow(chance/ (k2 + chance), self.expoente)
            if random() < chanceLargar: self.largar()
        self.passo(vizinhos)

    #DECISOES COM DADOS
    def distanciaEuclidiana(self, t1: tuple[int, int], t2: tuple[int, int] )-> float:
        return sqrt(pow(t1[0] -t2[0], 2) + pow(t1[1] - t2[1], 2))
    

    def similaridadeDaVizinhanca(self, item: Item, visao:list[Celula], alpha: float)->float:
        similaridade = 0
        for celula in visao:
            if celula.temItem():
                subtrai = self.distanciaEuclidiana(item.tupla(), celula.item.tupla())/alpha
                similaridade += 1.0 - subtrai
        fx = similaridade/len(visao)
        if fx < 0.0: fx = 0.0
        #print('f:', fx)
        return fx

    def tomadaDecisaoDados(self, vizinhos: list[Celula], visao: list[Celula], k1: int, k2: int, alpha: int):
        if (not self.carregaItem()) and self.posicao.temItem():
            chance = self.similaridadeDaVizinhanca(self.posicao.item,visao, alpha)
            chancePegar = pow(k1/ (k1 + chance), self.expoente)
            if (chancePegar > self.limitante) and (random() < chancePegar): self.pegar()
        elif self.carregaItem() and (not self.posicao.temItem()):
            chance = self.similaridadeDaVizinhanca(self.item, visao, alpha)
            chanceLargar = pow(chance/ (k2 + chance), 2)
            if (chanceLargar > self.limitante) and (random() < chanceLargar): self.largar()

        self.passo(vizinhos)

    def tomadaDecisaoDadosFinal(self, vizinhos: list[Celula], visao: list[Celula], k2: int, alpha: int):
        if self.carregaItem() and (not self.posicao.temItem()):
            chance = self.similaridadeDaVizinhanca(self.item, visao, alpha)
            chanceLargar = pow(chance/ (k2 + chance), 2)
            if random() < chanceLargar: self.largar()
        self.passo(vizinhos)


#AMBIENTE DE EXECUCAO
class Ambiente:
    ITERACOES = 'iteracoes'
    DIMENSAOX = 'dimensaoX'
    DIMENSAOY = 'dimensaoY'
    NUMERO_DE_AGENTES  = 'numero_de_agentes'
    VISAO = 'visao'
    ALPHA = 'alpha'
    K1 = 'k1'
    K2 = 'k2'
    LIMITANTE = 'limitante'

    def __init__(self,
                dimensoesAmbiente = (30,30),
                nAgentes = 10,
                alcanceVisaoAgente=1,
                expoente = 2,
                k1 = 0.1,
                k2 = 0.1,
                limitante = 0.01) -> None:


        self.nLinhas = dimensoesAmbiente[0]
        self.nColunas = dimensoesAmbiente[1]
        self.nAgentes = nAgentes
        self.listaItem = []
        self.alcanceVisaoAgente = alcanceVisaoAgente

        self.k1 = k1
        self.k2 = k2

        self.limitante = limitante
        self.alpha = 1

        self.listaAgente = [Agente(self.nColunas, self.nLinhas, visao=self.alcanceVisaoAgente, expoente=expoente, limitante = limitante) for i in range(self.nAgentes)]
        self.mapa = [[Celula(posicao=(i,j)) for j in range(self.nLinhas)] for i in range(self.nColunas)]
    

    def __init__(self, dicionario= {'dimensaoX':30, 'dimensaoY':30, 'numero_de_agentes':10, 'visao':1, 'alpha':1, 'k1': 0.5, 'k2': 0.5, 'limitante': 0.01}) -> None:
        self.nLinhas = dicionario[self.DIMENSAOX]
        self.nColunas = dicionario[self.DIMENSAOY]
        self.nAgentes = dicionario[self.NUMERO_DE_AGENTES]
        self.alcanceVisaoAgente = dicionario[self.VISAO]
        self.alpha = dicionario[self.ALPHA]
        self.k1 = dicionario[self.K1]
        self.k2 = dicionario[self.K2]
        self.limitante = dicionario[self.LIMITANTE]

        self.listaItem = []
        self.listaAgente = [Agente(self.nColunas, self.nLinhas, visao=self.alcanceVisaoAgente, limitante=self.limitante) for i in range(self.nAgentes)]
        self.mapa = [[Celula(posicao=(i,j)) for j in range(self.nLinhas)] for i in range(self.nColunas)]


    def carregaDados(self, listaItem: list[Item]):
        self.listaItem = listaItem
    
    
    def vizinhosRetangular(self, celula: Celula) -> list[Celula]:
    #funcao retorna uma lista com todas as celulas na vizinhança de visao.
        x = celula.x
        y = celula.y

        viz = []
        for i in range(max(0,x-1), min(self.nColunas, x+2)):
            for j in range(max(0,y-1), min(self.nLinhas,y+2)):
                viz.append(self.mapa[i][j])
        return viz
    
    def vizinhosCircular(self, celula: Celula, visao=1) -> list[Celula]:
    #funcao retorna uma lista com todas as celulas na vizinhança de visao.
        x = celula.x
        y = celula.y

        viz = []
        for i in range(x-visao, x+visao+1):
            for j in range(y-visao, y+visao+1):
                posx = i
                posy = j
                if (posx!=x) or (posy != y):
                    if i < 0:
                        posx = self.nColunas + i
                    if j < 0:
                        posy = self.nLinhas + j
                    if i >= self.nColunas:
                        posx = i - self.nColunas
                    if j >= self.nLinhas:
                        posy = j - self.nLinhas
                    viz.append(self.mapa[posx][posy])
        
        return viz
    


    def distanciaEuclidiana(self, c1: Celula, c2: Celula)->float:
        valor = pow(self.dX(c1.x, c2.x), 2) + pow(self.dY(c1.y,c2.y), 2)
        v = float(valor)
        return float(v)

    
    def inicia(self):
        #Agentes Iniciados 
        for agente in self.listaAgente:
            x = randrange(self.nColunas)
            y = randrange(self.nLinhas)
            while self.mapa[x][y].temAgente:
                            x = randrange(self.nColunas)
                            y = randrange(self.nLinhas)
            agente.iniciaAgente(self.mapa[x][y])
        print('Agentes Iniciados')

        #Itens Iniciados
        for item in self.listaItem:
            x = randrange(self.nColunas)
            y = randrange(self.nLinhas)
            if not self.mapa[x][y].temItem():
                self.mapa[x][y].setItem(item)
        print('Itens Iniciados')




    def iteracao(self):
        for agente in self.listaAgente:
            #agente.tomadaDecisaoComVisao(self.vizinhosCircular(agente.posicao,self.alcanceVisaoAgente), self.vizinhosCircular(agente.posicao))
            #agente.tomadaDecisaoIngenuaComVisao(self.vizinhosCircular(agente.posicao), self.vizinhosCircular(agente.posicao,agente.visao))
            #agente.tomadaDecisaoComFatoresK(self.vizinhosCircular(agente.posicao),self.vizinhosCircular(agente.posicao, agente.visao) , self.k1, self.k2)
            agente.tomadaDecisaoDados(self.vizinhosCircular(agente.posicao), self.vizinhosCircular(agente.posicao,  agente.visao), self.k1, self.k2, self.alpha)
    
    def iteracaoFinal(self)->bool:
        final = True
        for agente in self.listaAgente:
            #agente.tomadaDecisaoIngenuaFinal(self.vizinhosCircular(agente.posicao), self.vizinhosCircular(agente.posicao,agente.visao))
            #agente.tomadaDecisaoComFatoresFinal(self.vizinhosCircular(agente.posicao), self.vizinhosCircular(agente.posicao, agente.visao), self.k2)
            agente.tomadaDecisaoDadosFinal(self.vizinhosCircular(agente.posicao), self.vizinhosCircular(agente.posicao,  agente.visao), self.k2, self.alpha)
            if agente.carregaItem(): final=False
        return final
        
