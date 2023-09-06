from random import randrange, random
import pygame


class Celula:
    temAgente: bool
    temItem: bool
    x: int
    y: int
    def __init__(self,
                posicao: tuple[int,int],
                temAgente = False,
                temItem = False) -> None:
        self.temAgente = temAgente
        self.temItem = temItem
        self.agente = None
        self.x = posicao[0]
        self.y = posicao[1]
    
    def tupla(self)->tuple:
        return (self.x, self.y)


class Agente:
    carregaItem: bool
    visao: int
    posicao: Celula
    expoente: int
    def __init__(self,
                limX: int,
                limY: int,
                carregaItem=False,
                visao=1,
                expoente=2,
                limitante = 0.01) -> None:
        self.carregaItem = carregaItem
        self.visao = visao
        self.posicao = None

        self.limX = limX
        self.limY = limY
    
        self.expoente = expoente
        self.limitante = limitante

    def moveParaCelula(self, celula: Celula):
        self.posicao.temAgente = False
        self.posicao = celula
        self.posicao.temAgente = True

    def iniciaAgente(self, celula: Celula):
        self.posicao = celula
        self.posicao.temAgente = True
        

    def passo(self, vizinhos: list[Celula]):
        vizinhos_disponiveis = []
        for vizinho in vizinhos:
            if vizinho.temAgente is False:
                vizinhos_disponiveis.append(vizinho)
        if len(vizinhos_disponiveis)>0:
            escolha = randrange(0,len(vizinhos_disponiveis))
            self.moveParaCelula(vizinhos_disponiveis[escolha])

    def pegar(self):
        if self.posicao.temItem and (not self.carregaItem):
            self.posicao.temItem = False
            self.carregaItem = True
        else: raise Exception('Erro ao pegar')
    
    def largar(self):
        if (not self.posicao.temItem) and self.carregaItem:
            self.carregaItem = False
            self.posicao.temItem = True
        else: raise Exception('Erro ao largar')

    def fator(self, vizinhos: list[Celula]):
        itens = 0
        for vizinho in vizinhos:
            if vizinho.temItem:
                itens +=1
        return itens/len(vizinhos)

    def tomadaDecisaoIngenua(self, vizinhos: list[Celula]):
        chance = pow(self.fator(vizinhos),self.expoente)# se tem muitos, decisao a sim
        decisao = random() < chance
        if self.carregaItem and (not self.posicao.temItem) and decisao:
                self.largar()
        if (not self.carregaItem) and (self.posicao.temItem) and (not decisao):
                self.pegar()     
        self.passo(vizinhos)
    
    def tomadaDecisaoIngenuaComVisao(self, vizinhos: list[Celula], visao: list[Celula]):
        chance = pow(self.fator(visao),self.expoente)
        # se tem muitos, decisao a sim
        decisao = random() < chance
        if self.carregaItem and (not self.posicao.temItem) and decisao:
                self.largar()
        if (not self.carregaItem) and (self.posicao.temItem) and (not decisao):
                self.pegar()
        
        self.passo(vizinhos)
 
    def tomadaDecisaoIngenuaFinal(self, vizinhos: list[Celula], visao: list[Celula]):
        chance = pow(self.fator(visao),self.expoente)
        decisao = random() < chance
        if self.carregaItem and (not self.posicao.temItem) and decisao:
                self.largar()
        self.passo(vizinhos)

    def tomadaDecisaoComFatoresK(self, vizinhos: list[Celula], visao: list[Celula], k1: int, k2: int):
        if (not self.carregaItem) and self.posicao.temItem:
            chance = self.fator(visao)
            chancePegar = pow(k1/ (k1 + chance), self.expoente)
            if (chancePegar > self.limitante) and (random() < chancePegar): self.pegar()
        elif self.carregaItem and (not self.posicao.temItem):
            chance = self.fator(vizinhos)
            chanceLargar = pow(chance/ (k2 + chance), 2)
            if (chanceLargar > self.limitante) and (random() < chanceLargar): self.largar()

        self.passo(vizinhos)
    
    def tomadaDecisaoComFatoresFinal(self, vizinhos:list[Celula], visao: list[Celula], k2: int):
        if self.carregaItem and (not self.posicao.temItem):
            chance = self.fator(visao)
            chanceLargar = pow(chance/ (k2 + chance), self.expoente)
            if random() < chanceLargar: self.largar()
        self.passo(vizinhos)
    
    '''
    def dX(self, x1:int, x2:int):
        return min(abs(x1-x2), abs(x1 - (self.limX -x2)))
        
    def dY(self, y1:int, y2:int):
        return min(abs(y1-y2), abs(y1 -(self.limY - y2)))

    def distanciaEuclidiana(self, c1: Celula, c2: Celula)->float:
        valor = sqrt(pow(self.dX(c1.x,c2.x), 2) + pow(self.dY(c1.y,c2.y), 2))
        v = float(valor)
        return float(v)

    def similaridadeDaVizinhanca(self, raioDeVisao: int, fator:float, visao:list[Celula])->float:
        similaridade = 0
        for celula in visao:
            if celula.temItem:
                subtrai = self.distanciaEuclidiana(self.posicao, celula)/fator
                similaridade += 1.0 - subtrai
        fx = similaridade/pow(raioDeVisao,2)
        if fx < 0.0:
            return 0
        return fx

    def tomadaDecisaoComVisao(self, visao:list[Celula], vizinhos: list[Celula], k1=0.1, k2=0.1):
        fx = self.similaridadeDaVizinhanca(self.visao, self.visao, visao)
        #print(fx)
        if self.carregaItem and (not self.posicao.temItem):
            pd = pow(fx/(k2+fx), 2)
            decisao = random() < pd
            #print('largar:',decisao)
            if decisao:
                self.largar()
        if (not self.carregaItem) and (self.posicao.temItem):
            pp = pow((k1/(k1+ fx)), 2)
            decisao = random() < pp
            #print('pegar:',decisao)
            if decisao:
                self.pegar()

        self.passo(vizinhos)
    '''


class Ambiente:
    def __init__(self, 
                dimensoesAmbiente = (30,30),
                nAgentes = 10,
                nItens = 2,
                alcanceVisaoAgente=1,
                expoente = 2,
                k1 = 0.1,
                k2 = 0.1,
                limitante = 0.01) -> None:

        if nItens > dimensoesAmbiente[0]*dimensoesAmbiente[1]:
            raise Exception('Há mais itens que celulas')

        self.nLinhas = dimensoesAmbiente[0]
        self.nColunas = dimensoesAmbiente[1]
        self.nAgentes = nAgentes
        self.nItens = nItens
        self.alcanceVisaoAgente = alcanceVisaoAgente

        self.k1 = k1
        self.k2 = k2

        self.limitante = limitante

        self.vetorAgente = [Agente(self.nColunas, self.nLinhas, visao=self.alcanceVisaoAgente, expoente=expoente, limitante = limitante) for i in range(self.nAgentes)]
        self.mapa = [[Celula(posicao=(i,j)) for j in range(self.nLinhas)] for i in range(self.nColunas)]
    
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
        print(v)
        return float(v)

    
    def inicia(self):
        for agente in self.vetorAgente:
            x = randrange(self.nColunas)
            y = randrange(self.nLinhas)
            while self.mapa[x][y].temAgente:
                            x = randrange(self.nColunas)
                            y = randrange(self.nLinhas)
            agente.iniciaAgente(self.mapa[x][y])

        for item in range(self.nItens):
            x = randrange(self.nColunas)
            y = randrange(self.nLinhas)
            while self.mapa[x][y].temItem:
                x = randrange(self.nColunas)
                y = randrange(self.nLinhas)
            self.mapa[x][y].temItem = True


    def iteracao(self):
        for agente in self.vetorAgente:
            #agente.tomadaDecisaoComVisao(self.vizinhosCircular(agente.posicao,self.alcanceVisaoAgente), self.vizinhosCircular(agente.posicao))
            #agente.tomadaDecisaoIngenuaComVisao(self.vizinhosCircular(agente.posicao), self.vizinhosCircular(agente.posicao,agente.visao))
            agente.tomadaDecisaoComFatoresK(self.vizinhosCircular(agente.posicao),self.vizinhosCircular(agente.posicao, agente.visao) , self.k1, self.k2)
    
    def iteracaoFinal(self)->bool:
        final = True
        for agente in self.vetorAgente:
            #agente.tomadaDecisaoIngenuaFinal(self.vizinhosCircular(agente.posicao), self.vizinhosCircular(agente.posicao,agente.visao))
            agente.tomadaDecisaoComFatoresFinal(self.vizinhosCircular(agente.posicao), self.vizinhosCircular(agente.posicao, agente.visao), self.k2)
            if agente.carregaItem: final=False
        return final
        

        


class Exibicao:
    ambiente: Ambiente
    def __init__(self,
                ambiente: Ambiente(),
                resolucaoCelula = (20,20),
                corFundo = (50,50,50),
                corCelula = (200,200,200),
                corAgente = (255,0,0),
                corItem  = (0,0,100)) -> None:
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
        pygame.display.get_surface().fill(corFundo)

    def exibeRetangulo(self, x, y, cor):
        pygame.draw.rect(self.tela, cor, [x, y, self.dimensaoXCelula, self.dimensaoYCelula])

    def exibeCelula(self, x, y):
        #gera Celula
        posicaoX = x*self.dimensaoXCelula
        posicaoY = y*self.dimensaoYCelula

        #posicao da celula
        celula = self.ambiente.mapa[x][y]
        cor = self.corCelula
        if celula.temItem: cor = self.corItem

        pygame.draw.rect(self.tela, cor, [posicaoX+1, posicaoY+1, self.dimensaoXCelula -2, self.dimensaoYCelula -2])
            

    def exibeAgente(self, agente: Agente):
        posicaoX = agente.posicao.x * self.dimensaoXCelula
        posicaoY = agente.posicao.y * self.dimensaoYCelula
        pygame.draw.rect(self.tela, self.corAgente, [posicaoX +self.dimensaoXCelula/3, posicaoY +self.dimensaoYCelula/3, self.dimensaoXCelula/2, self.dimensaoYCelula/2])
        if agente.carregaItem:
            pygame.draw.rect(self.tela, self.corItem, [posicaoX + self.dimensaoXCelula/3, posicaoY + self.dimensaoYCelula/3, self.dimensaoXCelula/3, self.dimensaoYCelula/3])


    def exibeFrame(self):
        for x in range(self.ambiente.nLinhas):
            for y in range(self.ambiente.nColunas):

                self.exibeCelula(x,y)
        for agente in self.ambiente.vetorAgente:
            self.exibeAgente(agente)

        pygame.display.update()


