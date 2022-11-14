from leitor import Leitor
from ambiente import Ambiente
from exibicao import Exibicao
from indice import Indice
import matplotlib.pyplot as plt


class Control:
    leitor : Leitor
    ambiente : Ambiente
    exibicao : Exibicao
    ambientes : list[Ambiente]

    def __init__(self, databasePath: str, configPath:str) -> None:
        self.leitor = Leitor()
        dados = self.leitor.carregaDatabase(databasePath)
        configuracoes = self.leitor.carregaConfiguracoes(configPath)
        print('CONFIGURACOES = ',configuracoes)

        self.ambiente = Ambiente(dados,configuracoes)
        self.ambiente.iniciaMatriz()
        self.ambiente.iniciacaminhoAtual()
        self.ambiente.set_cooling()

        self.ambientes = [Ambiente(dados,configuracoes) for i in range(10)]
        for amb in self.ambientes:
            amb.iniciaMatriz()
            amb.iniciacaminhoAtual()
            amb.set_cooling()

    def start(self):
        i = 0
        while self.ambiente.iteracao()==True:
            i = 0
        self.ambiente.exibir()

    def boxplot(self):
        l_box_plot = []
        for teste in self.ambientes:
            while teste.iteracao()==True:
                pass
            l_box_plot.append(teste.pesocaminhoAtual)
        
        plt.boxplot(l_box_plot)
        plt.title('Boxplot Peso dos Caminhos')
        plt.show()


        


