import random

class Problem:
    def encode(self,solution:list)->list:...

    def decode(self,cromossomo:list)->list:...
    
    def fitness(self,solution:list)->any:...

    def set_problem(self,config:dict)->dict:...


class Ambiente:
    type_dict = {
        'BIN': bool,
        'INT': int,
        'INT-PERM':int,
        'REAL': float
    }
    
    def __init__(self,config:dict,problem:Problem) -> None:
        self.problem = problem
        self.config = problem.set_problem(config)
        self.population = self.generate_population()
        

    def generate_population(self):
        pop = int(self.config['POP'])
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
                bound = list(map(int,self.config['BOUND'].strip('][ ').split(',')))
                individuo = random.sample(range(*bound),k=dim)
                assert len(set(individuo)) == len(individuo)
                return individuo
            
            case 'REAL':
                bound = list(map(int,self.config['BOUND'].strip('][ ').split(',')))
                return [random.random()*(bound[1]-bound[0])+bound[0] for _ in range(dim)]
            case _:
                return random.choices((1,0),k=dim)
    

            

