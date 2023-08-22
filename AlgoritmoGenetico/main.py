from scripts.leitor import Leitor
from scripts.ambiente import Ambiente
from scripts.problems import Nrainhas


leitor = Leitor()
nrainhas = Nrainhas()


ambiente = Ambiente(config=leitor.loadConfig("AlgoritmoGenetico/data/data_nrainhas8.txt"),problem=nrainhas)
print('config:',ambiente.config)
print('pop:',ambiente.population)
print('instance:',[nrainhas.decode(cromossomo) for cromossomo in ambiente.population])
print(nrainhas._diagonal)