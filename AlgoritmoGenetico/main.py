from scripts.leitor import Leitor
from scripts.ambiente import Ambiente
from scripts.problems import Nrainhas


leitor = Leitor()
nrainhas = Nrainhas()

#Carregando problema
ambiente = Ambiente(config=leitor.loadConfig("AlgoritmoGenetico/data/data_nrainhas8.txt"),problem=nrainhas)
print('config:',ambiente.config)
print('pop:',ambiente.population)
print('instance:',[nrainhas.decode(cromossomo) for cromossomo in ambiente.population])

#Definindo melhor e pior solução
ambiente.evaluate()
best = ambiente.population[ambiente.population.index(max(ambiente.population))]
worst = ambiente.population[ambiente.population.index(min(ambiente.population))]

print('best:\n',nrainhas.get_matrix(nrainhas.decode(best)))

print('worst:\n',nrainhas.get_matrix(nrainhas.decode(worst)))

