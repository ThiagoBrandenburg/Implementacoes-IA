# from utils import *
import sys
import copy
import random
import matplotlib.pyplot as plt

# VARIÁVEIS DE CONTROLE
chance_crossover = int(sys.argv[1])
chance_mutacao = int(sys.argv[2])
vetor_crossover = [1] * chance_crossover + [0] * (100 - chance_crossover)
vetor_mutacao = [1] * chance_mutacao + [0] * (1000 - chance_mutacao)
N = 8  # número de rainhas
C_MIN = N * (N - 1)  # todas as rainhas colidindo com todas as rainhas menos ela mesma
ELITISMO = 0
melhor_fit_vetor = []
pior_fit_vetor = []
media_fit_vetor = []
melhor_ind_vetor = []


def check_colisao(linha_n, coluna_n, rainhas):
    colisoes = 0
    diagonais = []
    d1 = d2 = d3 = d4 = (linha_n, coluna_n)
    # PEGA TODAS AS DIAGONAIS
    for i in range(N):
        d1 = (d1[0] + 1, d1[1] + 1)
        d2 = (d2[0] - 1, d2[1] + 1)
        d3 = (d3[0] + 1, d3[1] - 1)
        d4 = (d4[0] - 1, d4[1] - 1)
        if d2[0] >= 0 and d3[1] >= 0 and d2[1] < N and d3[0] < N:
            diagonais.append(d2)
            diagonais.append(d3)
        if d4[0] >= 0:
            diagonais.append(d4)
        if d1[0] < N:
            diagonais.append(d1)
    # CHECA SE EXISTEM RAINHAS NESSES PONTOS
    for diagonal in diagonais:
        linha = diagonal[0]
        coluna = diagonal[1]
        if rainhas[linha] == coluna:
            colisoes += 1
    return colisoes


def conta_colisao(tabuleiro):
    colisoes = []
    for index, individuo in enumerate(tabuleiro):
        colisoes.append(check_colisao(index, individuo, tabuleiro))
    total_colisoes = sum(colisoes)
    return total_colisoes


# FUNÇÃO FITNESS
def funcao_fitness(populacao):
    fitness = []
    for tabuleiro in populacao:
        total_colisoes = conta_colisao(tabuleiro)
        if C_MIN - total_colisoes > 0:
            fitness.append(C_MIN - total_colisoes)
        else:
            fitness.append(0)
    return fitness


def get_melhor_individuo(vetor_fitness, populacao):
    melhor_fit = max(vetor_fitness)
    melhor_index = vetor_fitness.index(melhor_fit)
    melhor = copy.deepcopy(populacao[melhor_index])
    return melhor, melhor_index, melhor_fit


def get_pior_individuo(vetor_fitness, populacao):
    pior_fit = min(vetor_fitness)
    pior_index = vetor_fitness.index(pior_fit)
    pior = populacao[pior_index]
    return pior, pior_index, pior_fit


def cycle_crossover(p1, p2):
    # faz o crossover ou não
    # pesos = [1] * chance_crossover + [0] * (100 - chance_crossover)
    pesos = vetor_crossover
    muda = random.choice(pesos)
    if muda:
        item = p2[0]
        for i in range(1, len(p1)):
            if p1[i] == item:
                item = p2[i]
            else:
                p1[i], p2[i] = p2[i], p1[i]
    return p1, p2


def mutacao(individuo):
    for i in range(len(individuo)):
        # faz a mutação ou não
        # pesos = [1] * chance_mutacao + [0] * (10000 - chance_mutacao)
        pesos = vetor_mutacao
        muda = random.choice(pesos)
        if muda == 1:
            posicao_mudar = random.randint(0, N - 1)
            individuo[i], individuo[posicao_mudar] = (
                individuo[posicao_mudar],
                individuo[i],
            )
    return individuo


def selecao_proporcional(populacao, fitness):
    # Calcula a soma total dos valores de fitness
    soma_fitness = sum(fitness)

    # Gere um número aleatório entre 0 e a soma total de fitness
    valor_aleatorio = random.uniform(0, soma_fitness)

    # Inicializa uma variável para rastrear a soma cumulativa do fitness
    soma_cumulativa = 0

    # Itera sobre a população para encontrar o indivíduo selecionado
    for i, ind in enumerate(populacao):
        soma_cumulativa += fitness[i]

        # Se a soma cumulativa exceder o valor aleatório, selecione este indivíduo
        if soma_cumulativa >= valor_aleatorio:
            return ind

    # Se não encontrar nenhum indivíduo (o que é improvável), retorne o último
    return populacao[-1]


def substitui_pior_por_melhor(populacao, pior_index, melhor):
    populacao[pior_index] = melhor
    return populacao


def get_populacao_int_permutado(file_name):
    linhas = []
    f = open(f"populacoes/{file_name}.txt")
    for linha in f.readlines():
        linha = linha.replace("\n", "")
        linhas.append([int(x) for x in linha.split(" ")])
    f.close()
    return linhas


def pipeline_evolutiva(populacao_inicial, epochs=100):
    # media_fit_vetor_local, melhor_fit_vetor_local = [], []
    for i in range(epochs):
        fitness = funcao_fitness(populacao_inicial)
        melhor, index_melhor, melhor_fit = get_melhor_individuo(
            fitness, populacao_inicial
        )
        media_fit_vetor.append(sum(fitness) / len(fitness))
        melhor_fit_vetor.append(melhor_fit)

        populacao_mutada = []
        for l in range(len(populacao_inicial)):
            populacao_mutada.append(selecao_proporcional(populacao_inicial, fitness))
        populacao_inicial = populacao_mutada
        populacao_mutada = []
        for k in range(0, len(populacao_inicial) - 1, 2):
            p1, p2 = cycle_crossover(populacao_inicial[k], populacao_inicial[k + 1])
            populacao_mutada.append(p1)
            populacao_mutada.append(p2)
        populacao_inicial = populacao_mutada

        populacao_mutada = []
        for k in range(len(populacao_inicial)):
            p1 = mutacao(populacao_inicial[k])
            populacao_mutada.append(p1)
        populacao_inicial = populacao_mutada

        fitness = funcao_fitness(populacao_inicial)
        pior, index_pior, pior_fit = get_pior_individuo(fitness, populacao_inicial)
        populacao_inicial = substitui_pior_por_melhor(
            populacao_inicial, index_pior, melhor
        )
    # media_fit_vetor.append(media_fit_vetor_local)
    # melhor_fit_vetor.append(melhor_fit_vetor_local)


def plotar_grafico_de_linhas(
    vetor1,
    vetor2,
    titulo="Gráfico de Linhas",
    rotulo_x="X",
    rotulo_y="Y",
    legenda1="Linha 1",
    legenda2="Linha 2",
    salvar_imagem=True,  # Adicionamos um novo parâmetro para controlar o salvamento da imagem
    nome_imagem="grafico.png",  # Nome da imagem a ser salva
):
    plt.figure(figsize=(8, 6))
    plt.plot(vetor1, label=legenda1)
    plt.plot(vetor2, label=legenda2)
    plt.title(titulo)
    plt.xlabel(rotulo_x)
    plt.ylabel(rotulo_y)
    plt.legend()
    plt.grid(True)

    if salvar_imagem:
        plt.savefig(f"output/{nome_imagem}", bbox_inches="tight")  # Salva a imagem

    plt.show()

    # def plotar_grafico_de_linhas(
    #     vetor1,
    #     vetor2,
    #     titulo="Gráfico de Linhas",
    #     rotulo_x="X",
    #     rotulo_y="Y",
    #     legenda1="Linha 1",
    #     legenda2="Linha 2",
    # ):
    #     """
    #     Plota um gráfico de duas linhas com base em dois vetores fornecidos.

    #     Parâmetros:
    #     - vetor1: Primeiro vetor de valores a serem plotados.
    #     - vetor2: Segundo vetor de valores a serem plotados.
    #     - titulo: Título do gráfico (opcional, padrão é "Gráfico de Linhas").
    #     - rotulo_x: Rótulo do eixo x (opcional, padrão é "X").
    #     - rotulo_y: Rótulo do eixo y (opcional, padrão é "Y").
    #     - legenda1: Legenda para a primeira linha (opcional, padrão é "Linha 1").
    #     - legenda2: Legenda para a segunda linha (opcional, padrão é "Linha 2").
    #     """
    #     plt.figure(figsize=(8, 6))  # Define o tamanho da figura (opcional)

    #     # Plota as duas linhas com legendas
    #     plt.plot(vetor1, label=legenda1)
    #     plt.plot(vetor2, label=legenda2)

    #     # Define o título e rótulos dos eixos
    #     plt.title(titulo)
    #     plt.xlabel(rotulo_x)
    #     plt.ylabel(rotulo_y)

    #     # Adiciona uma legenda ao gráfico
    #     plt.legend()

    #     # Exibe o gráfico
    #     plt.grid(True)  # Adiciona uma grade de fundo (opcional)
    plt.show()


pop = get_populacao_int_permutado(f"C:\Users\thiag\Documents\UDESC\Laboratorio\Implementacoes\Implementacoes-IA\AlgoritmoGenetico\populacoes\pop_nrainhas_8.txt")
pipeline_evolutiva(pop)
# print(melhor_fit_vetor[-1], chance_crossover, chance_mutacao)
for x in media_fit_vetor:
    print(x, end=" ")
print()
for x in melhor_fit_vetor:
    print(x, end=" ")
print()
print(melhor_fit_vetor[-1])

plotar_grafico_de_linhas(
    media_fit_vetor,
    melhor_fit_vetor,
    legenda1="Média Fitness",
    legenda2="Melhor Fitness",
    salvar_imagem=True,  # Ativa o salvamento da imagem
    nome_imagem="grafico_nrainhas_{N}.png",  # Nome da imagem a ser salva
)
