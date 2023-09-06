from leitor import Leitor
from ambiente import Ambiente
from exibicao import Exibicao
from indice import Indice
import matplotlib.pyplot as plt


class Control:
    leitor : Leitor
    ambiente : Ambiente
    exibicao : Exibicao
    ambientes : list[list[Ambiente]]
    grafico : str

    def __init__(self, configPath:str) -> None:
        
        self.leitor = Leitor()
        configuracoes = self.leitor.carregaConfiguracoes(configPath)
        self.grafico = configuracoes[Indice.GRAFICO.value]
        print('CONFIGURACOES = ',configuracoes)
        
        
        databasePath = configuracoes[Indice.PATHDATABASE.value]
        dados = self.leitor.carregaDatabase(databasePath)
        self.ambiente = Ambiente(dados,configuracoes)
        self.ambiente.iniciaMatriz()
        self.ambiente.iniciacaminhoAtual()
        self.ambiente.set_cooling()
        
        if configuracoes[Indice.GRAFICO] == 'boxplot':
            self.ambientes = [[],[],[]]
            #c0
            configuracoes[Indice.COOLING] == '0'
            self.ambientes[0] = [Ambiente(dados,configuracoes) for i in range(10)]
            for amb in self.ambientes[0]:
                amb.iniciaMatriz()
                amb.iniciacaminhoAtual()
                amb.set_cooling()
            #c1
            configuracoes[Indice.COOLING] == '1'
            self.ambientes[1] = [Ambiente(dados,configuracoes) for i in range(10)]
            for amb in self.ambientes[1]:
                amb.iniciaMatriz()
                amb.iniciacaminhoAtual()
                amb.set_cooling()
            #c5
            configuracoes[Indice.COOLING] == '5'
            self.ambientes[2] = [Ambiente(dados,configuracoes) for i in range(10)]
            for amb in self.ambientes[2]:
                amb.iniciaMatriz()
                amb.iniciacaminhoAtual()
                amb.set_cooling()

 

    def start(self):
        if self.grafico == 'padrao':
            self.graph()
        elif self.grafico == 'boxplot':
            self.boxplot()
        else:
            print('(control) Tipo de grafico nao informado')


    def graph(self):
        i = 0
        while self.ambiente.iteracao()==True:
            i = 0
        self.ambiente.exibir()

    def boxplot(self):
        l_box_plot = [[],[],[]]
        for i in range(3):
            for teste in self.ambientes[i]:
                while teste.iteracao()==True:
                    pass
                l_box_plot[i].append(teste.pesocaminhoAtual)
            #print(min(l_box_plot))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(l_box_plot)
        #plt.boxplot(l_box_plot)
        ax.set_xticklabels(['cooling0','cooling1','cooling5'])
        plt.title('Boxplot Peso dos Caminhos')
        plt.show()
        