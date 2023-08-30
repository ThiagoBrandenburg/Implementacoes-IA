from scripts.problems import FabricaDeRadios
from scripts.ambiente import Ambiente
from scripts.leitor import Leitor

leitor = Leitor()
#config = leitor.loadConfig('AlgoritmoGenetico\data\data_algebric.txt')
radio = FabricaDeRadios()
config = leitor.loadConfig('AlgoritmoGenetico/data/data_fabrica_de_radios.txt')
print('printa:',config)
ambiente = Ambiente(config,FabricaDeRadios())
ambiente.generate_population()
ambiente.evaluate()

best = radio.decode(ambiente.population[ambiente.evaluation.index(max(ambiente.evaluation))])
worst = radio.decode(ambiente.population[ambiente.evaluation.index(min(ambiente.evaluation))])

print('Best:',radio.decode(best))
print('Worst:',radio.decode(worst))
print('Elite:',ambiente.elite_population)
print('Fit_max(best):',radio.fit_max(best))
print('Fit_min(best):',radio.fit_min(best))
print('Fit_max(worst):',radio.fit_max(worst))
print('Fit_min(worst):',radio.fit_min(worst))