from random import seed
import ambiente as ab
import pygame

#Formigueiro
formigueiro = (40,40)
formigas = 20
formigasMortas = 300
visao = 1
expoente = 2
#pegar
k1 = 0.1
#largar
k2 = 0.6
#limitante de eventos 
limite = 0.05

'''
Comportamentos bons
k1,k2 = (0.3, 0.5), (0.1, 0.6)<- esse é very bom
'''

#Tela
resolucaoCelula = (20,20)

#Iteracoes
iteracoes = 50000

seed()

ambiente = ab.Ambiente(dimensoesAmbiente=formigueiro, 
                        nAgentes=formigas,
                        nItens=formigasMortas,
                        alcanceVisaoAgente=visao,
                        expoente = expoente,
                        k1 = k1,
                        k2 = k2)
ambiente.inicia()


exibicao = ab.Exibicao(ambiente=ambiente, resolucaoCelula=resolucaoCelula)


for i in range(iteracoes):
    ambiente.iteracao()
    exibicao.exibeFrame()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
print('Threshold passado')
while not ambiente.iteracaoFinal():
    exibicao.exibeFrame()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
exibicao.exibeFrame()

input('Pressione para terminar')