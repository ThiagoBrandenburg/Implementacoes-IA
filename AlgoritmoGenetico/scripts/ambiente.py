import random
import numpy as np
from types import NoneType

class Problem:
    def encode(self,solution)->np.array:...

    def decode(self,cromossomo:np.array)->any:...

    def objective_function(self,solution)->any:...
    
    def fitness(self,solution)->any:...

    def set_problem(self,config:dict)->dict:...

    def generate_population(self,pop_size)->list:...

    def fit_max(self,solution)->any:...

    def fit_min(self,solution)->any:...


class Ambiente:
    type_dict = {
        'BIN': bool,
        'INT': int,
        'INT-PERM':int,
        'REAL': float
    }
    
    def __init__(self,config:dict,problem:Problem) -> None:
        random.seed()

        self.problem = problem
        self.config = config
        self.config = problem.set_problem(config)
        self.pop_size = int(self.config['POP'])
        self.dim_size = int(self.config['DIM'])
        self.population = self.generate_population()
        self.evaluation = self.evaluate(self.population)
        self.elitism = int(self.config['ELITISM']) if 'ELITISM' in self.config.keys() else 1
        self.mutation_rate = float(self.config['MUTATION_RATE']) if 'MUTATION_RATE' in self.config.keys() else 0.05
        self.elite_population = []
        self.elite_evaluation = []
        self.mating_pool = []

        self.results_best = []
        self.results_mean = []

        self.mutation_rate = 0.05

    def generate_population(self):
        if self.config['COD'] == 'CUSTOM':
            return self.problem.generate_population(self.pop_size)
        else:
            population = [self.gerar_individuo() for _ in range(self.pop_size)]
            return population
    
    def gerar_individuo(self):
        dim = int(self.config['DIM'])
        match self.config['COD']:
            case 'BIN':
                individuo = np.array(random.choices((1,0),k=dim))
                return individuo
            case 'INT':
                bound = list(map(int,self.config['BOUND'].strip('][ ').split(',')))
                individuo = np.array([random.randint(*bound) for _ in range(dim)])
                return individuo
            case 'INT-PERM':
                bound = (0,dim)
                individuo = random.sample(range(*bound),k=dim)
                assert len(set(individuo)) == len(individuo)
                return np.array(individuo)
            case 'REAL':
                bound = list(map(int,self.config['BOUND'].strip('][ ').split(',')))
                individuo = np.array([random.random()*(bound[1]-bound[0])+bound[0] for _ in range(dim)])
                return individuo
            case _:
                return random.choices((1,0),k=dim)
        
    def evaluate(self,population=None)->np.ndarray:
        '''Avalia uma população, caso não seja fornecido uma população como argumento, avalia self.population e atualiza/retorna self.evaluation'''
        if type(population) == NoneType:
            evaluation = np.array([self.problem.fitness(self.problem.decode(cromossomo)) for cromossomo in self.population])
            self.evaluation = evaluation
            return evaluation
        else:
            evaluation =  np.array([self.problem.fitness(self.problem.decode(cromossomo)) for cromossomo in population])
            return evaluation

    def save_elite(self):
        evaluation_positions_sorted = sorted(range(len(self.evaluation)),key=lambda x: self.evaluation[x],reverse=True)
        self.elite_population = [self.population[evaluation_positions_sorted[i]] for i in range(self.elitism)]
        self.elite_evaluation = [self.evaluation[evaluation_positions_sorted[i]] for i in range(self.elitism)]

    def _p_selection(self,evaluation):
        soma = sum(evaluation)
        p_selecao = [x/soma for x in evaluation]
        p_selecao = list(enumerate(p_selecao))
        p_selecao = sorted(p_selecao,key=lambda x: x[1])
        prob = [p_selecao[0]]
        for i in range(1,len(p_selecao)):
            el = (p_selecao[i][0], p_selecao[i][1] +prob[i-1][1])
            prob.append(el)
        chance = random.random()
        #print('prob:',prob)
        for elemento in prob:
            if chance < elemento[1]:
                return elemento[0]
            

    def roulette_wheel(self):
        #print('evaluation:',self.evaluation)
        e0 = self._p_selection(self.evaluation)
        replacement = self.evaluation.copy()
        #print('e0:',e0)
        replacement = np.delete(self.evaluation,e0,0)
        #print('replacement:',replacement)
        e1 = self._p_selection(replacement)
        e1 = e1 if e1 < e0 else e1+1
        return (self.population[e0],self.population[e1])
    
    def generate_mating_pool(self):
        '''Gera a Mating pool com base na população'''
        mating_pool = np.array([self.roulette_wheel()[i] for i in range(0,2) for _ in range(self.pop_size//2)])
        self.mating_pool = mating_pool
        return mating_pool


    def _mutate(self,cromossomo:np.ndarray):
        for gene in cromossomo:
            chance = random.random()
            if chance <= self.mutation_rate:
                #print('mutou!')
                match self.config['COD']:
                    case 'BIN':
                        gene = not gene
                    case 'INT':
                        cr_std = cromossomo.std()
                        print('cr_std',cr_std)
                        gene = gene + random.randint(-cr_std,cr_std)
                    case 'INT-PERM':
                        bound = (0,self.dim_size)
                        individuo = random.sample(range(*bound),k=self.dim_size)
                        assert len(set(individuo)) == len(individuo)
                        return individuo
                    case 'REAL':
                        bound = list(map(int,self.config['BOUND'].strip('][ ').split(',')))
                        return [random.random()*(bound[1]-bound[0])+bound[0] for _ in range(self.dim_size)]
                    case _:
                        return random.choices((1,0),k=self.dim_size)
    
    def generate_mutation(self,population):
        mutated_population = [self._mutate(cromossomo) for cromossomo in population]
        return mutated_population
                    
    def loop(self):
        print('Loop()')
        self.save_elite()


        self.generate_mating_pool()
        self.generate_mutation(self.mating_pool)
        mating_pool_evaluation = self.evaluate(self.mating_pool)
        for elite,elite_eval in zip(self.elite_population,self.elite_evaluation):
            pior = mating_pool_evaluation.argmin()
            self.mating_pool[pior] = elite
            mating_pool_evaluation[pior] = elite_eval
        self.population = self.mating_pool
        self.evaluation = mating_pool_evaluation
        
        #Add Results
        self.results_best.append(self.evaluation.max())
        self.results_mean.append(self.evaluation.mean())

        print('Elite eval:',self.elite_evaluation)
        #end of execution