import random

class Problem:
    def encode(self,solution:list)->list:...

    def decode(self,cromossomo:list)->list:...
    
    def fitness(self,solution:list)->any:...

    def set_problem(self,config:dict)->dict:...

    def generate_population(self,pop_size)->list:...


class Ambiente:
    type_dict = {
        'BIN': bool,
        'INT': int,
        'INT-PERM':int,
        'REAL': float
    }
    
    def __init__(self,config:dict,problem:Problem) -> None:
        self.problem = problem
        self.config = config
        self.config = problem.set_problem(config)
        self.population = self.generate_population()
        self.evaluation = [0 for _ in range(len(self.population))]
        self.elitism = int(self.config['ELITISM']) if 'ELITISM' in self.config.keys() else 1
        self.elite_population = []

        

    def generate_population(self):
        pop = int(self.config['POP'])
        if self.config['COD'] == 'CUSTOM':
            return self.problem.generate_population(pop)
        else:
            population = [self.gerar_individuo() for _ in range(pop)]
            return population
    
    def gerar_individuo(self):
        dim = int(self.config['DIM'])
        match self.config['COD']:
            case 'BIN':
                return random.choices((1,0),k=dim)
            case 'INT':
                bound = list(map(int,self.config['BOUND'].strip('][ ').split(',')))
                return [random.randint(*bound) for _ in range(dim)]
            case 'INT-PERM':
                bound = (0,dim)
                individuo = random.sample(range(*bound),k=dim)
                assert len(set(individuo)) == len(individuo)
                return individuo
            case 'REAL':
                bound = list(map(int,self.config['BOUND'].strip('][ ').split(',')))
                return [random.random()*(bound[1]-bound[0])+bound[0] for _ in range(dim)]
            case _:
                return random.choices((1,0),k=dim)
        
    def evaluate(self):
        '''Avalia as alternativas e salva no vetor evaluation'''
        self.evaluation = [self.problem.fitness(self.problem.decode(cromossomo)) for cromossomo in self.population]

    def save_elite(self):
        evaluation_positions_sorted = sorted(range(len(self.evaluation)),key=lambda x: self.evaluation[x],reverse=True)
        self.elite_population = [self.population[evaluation_positions_sorted[i]] for i in range(self.elitism)]