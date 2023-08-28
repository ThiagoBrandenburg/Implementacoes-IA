from scripts.problems import AlgebricFunction
from scripts.leitor import Leitor

leitor = Leitor()
#config = leitor.loadConfig('AlgoritmoGenetico\data\data_algebric.txt')
config = leitor.loadConfig('AlgoritmoGenetico/data/data_algebric.txt')
print('printa:',config)
af = AlgebricFunction(config,precision=0.001)
numero = 1.2978
print('numero=',numero)
print('encode(numero)=',af.encode(numero))
print('decode(encode(numero))=',af.decode(af.encode(numero)))
print('fit_max(numero)=',af.fit_max(numero))
print('fit_min(numero)=',af.fit_min(numero))