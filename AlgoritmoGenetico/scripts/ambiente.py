import random
import numpy as np
from types import NoneType
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools 

class Problem:
    def encode(self, solution) -> np.array:
        ...

    def decode(self, cromossomo: np.array) -> any:
        ...

    def objective_function(self, solution) -> any:
        ...

    def penality_function(self, solution) -> any:
        ...

    def fitness(self, solution) -> any:
        ...

    def set_problem(self, config: dict) -> dict:
        ...

    def generate_population(self, pop_size) -> list:
        ...

    def fit_max(self, solution) -> any:
        ...

    def fit_min(self, solution) -> any:
        ...


class Ambiente:
    type_dict = {"BIN": bool, "INT": int, "INT-PERM": int, "REAL": float}

    def __init__(self, 
                 config: dict, 
                 problem: Problem,
                 parallel=False) -> None:
        
        random.seed()
        #Declaração das variáveis
        self.counter = 0
        self.parallel = parallel
        self.problem = problem
        self.config = config
        self.config = problem.set_problem(config)
        self.pop_size = int(self.config["POP"])
        self.dim_size = int(self.config["DIM"])
        self.bound_size = eval(self.config["BOUND"])
        self.elitism = (
            int(self.config["ELITISM"]) if "ELITISM" in self.config.keys() else 1
        )
        self.mutation_rate = (
            float(self.config["MUTATION_RATE"])
            if "MUTATION_RATE" in self.config.keys()
            else 0.05
        )
        self.cross_over_rate = (
            (float(self.config["CROSSOVER_RATE"]))
            if "CROSSOVER_RATE" in self.config.keys()
            else 0.8
        )
        self.iterations = (
            int(self.config["ITERATIONS"])
            if "ITERATIONS" in self.config.keys()
            else 100
        )
        self.save_penality = (
            bool(self.config["SAVE_PENALITY"])
            if "SAVE_PENALITY" in self.config.keys()
            else False         
        )
        self.win_rate = (
            float(self.config['WIN_RATE'])
            if 'WIN_RATE' in self.config.keys()
            else 0.9
        )
        self.dizimate = (
            bool(self.config['DIZIMATE'])
            if 'DIZIMATE' in self.config.keys()
            else True
        )
        self.dizimation_interval = (
            int(self.config['DIZIMATION_INTERVAL'])
            if 'DIZIMATION_INTERVAL' in self.config.keys()
            else 100
        )


        #Geração da população inicial
        self.population = self.generate_population()
        self.evaluation = self.evaluate(self.population)

        #Resultados
        self.elite_population = []
        self.elite_evaluation = []
        self.mating_pool = []
        self.results_best = []
        self.results_mean = []
        self.results_penality = []



    def generate_population(self):
        if self.config["COD"] == "CUSTOM-INT":
            return self.problem.generate_population(self.pop_size)
        else:
            population = [
                self.gerar_individuo(individuo) for individuo in range(self.pop_size)
            ]
            return population

    def gerar_individuo(self, pos: int):
        dim = int(self.config["DIM"])
        match self.config["COD"]:
            case "BIN":
                individuo = np.array(random.choices((1, 0), k=dim))
                return individuo
            case "INT":
                # bound = list(map(int, self.config["BOUND"].strip("][ ").split(",")))
                # bound = eval(self.config['BOUND'])
                # print('(gr_indv)','bound:',self.bound_size,'pos:',pos,'pop_size',self.pop_size)
                individuo = np.array(
                    [
                        random.randint(self.bound_size[i][0], self.bound_size[i][1] - 1)
                        for i in range(dim)
                    ]
                )
                return individuo
            case "INT-PERM":
                # bound = (0, dim)
                #print(self.bound_size)
                individuo = random.sample(
                    range(self.bound_size[0][0], self.bound_size[0][1]), k=dim
                )
                assert len(set(individuo)) == len(individuo)
                return np.array(individuo)
            case "REAL":
                # bound = list(map(int, self.config["BOUND"].strip("][ ").split(",")))
                individuo = np.array(
                    [
                        random.random()
                        * (self.bound_size[pos][1] - self.bound_size[pos][0])
                        + self.bound_size[pos][0]
                        for _ in range(dim)
                    ]
                )
                return individuo
            case _:
                return random.choices((1, 0), k=dim)

    def evaluate(self, population) -> np.ndarray:
        """Avalia uma população com problem.fitness"""
        if self.parallel is False:
            evaluation = np.array(
                [
                    self.problem.fitness(self.problem.decode(cromossomo))
                    for cromossomo in population
                ]
            )
        else:
            # def fx(x):
            #     return self.problem.fitness()
            evaluation = np.array(
                Parallel(n_jobs=-1)(
                    delayed(self.problem.fitness)(self.problem.decode(cromossomo))
                    for cromossomo in population
                )
            )
        return evaluation

    def save_elite(self):
        '''Infelizmente temos que salvar a elite economica deste algoritmo :'( '''
        evaluation_positions_sorted = sorted(
            range(len(self.evaluation)), key=lambda x: self.evaluation[x], reverse=True
        )
        self.elite_population = [
            self.population[evaluation_positions_sorted[i]] for i in range(self.elitism)
        ]
        self.elite_evaluation = [
            self.evaluation[evaluation_positions_sorted[i]] for i in range(self.elitism)
        ]

    def _p_selection(self, evaluation):
        soma = sum(evaluation)
        p_selecao = [x / soma for x in evaluation]
        p_selecao = list(enumerate(p_selecao))
        p_selecao = sorted(p_selecao, key=lambda x: x[1])
        prob = [p_selecao[0]]
        for i in range(1, len(p_selecao)):
            el = (p_selecao[i][0], p_selecao[i][1] + prob[i - 1][1])
            prob.append(el)
        chance = random.random()
        # print('prob:',prob)
        for elemento in prob:
            if chance < elemento[1]:
                return elemento[0]

    def roulette_wheel(self):
        # print('evaluation:',self.evaluation)
        e0 = self._p_selection(self.evaluation)
        replacement = self.evaluation.copy()
        # print('e0:',e0)
        replacement = np.delete(self.evaluation, e0, 0)
        # print('replacement:',replacement)
        e1 = self._p_selection(replacement)
        e1 = e1 if e1 < e0 else e1 + 1
        return (self.population[e0], self.population[e1])

    def _estocastic_tournament(self, sample_size=2, win_rate=1.0):
        participants = random.sample(range(self.pop_size), sample_size)
        gene = self.evaluation[participants].argmax()
        if random.random() > win_rate:
            gene = self.evaluation[participants].argmin()
        cromossomo = self.population[gene]
        return cromossomo

    def generate_mating_pool(self):
        """Gera a Mating pool com base em self.population"""
        mating_pool = np.array(
            [self._estocastic_tournament(win_rate=self.win_rate) for _ in range(self.pop_size)]
            #[cromossomo for cromossomo in self.roulette_wheel() for _ in range(self.pop_size//2)]
        )
        return mating_pool

    def _mutate(self, cromossomo: np.ndarray):
        for gene in range(self.dim_size):
            chance = random.random()
            if chance <= self.mutation_rate:
                alelo = cromossomo[gene]
                match self.config["COD"]:
                    case "BIN":
                        cromossomo[gene] = not alelo
                    case "INT":
                        # cr_std = round(cromossomo.std())
                        # print("cr_std", cr_std)
                        # cromossomo[gene] = alelo + random.randint(-cr_std, cr_std)
                        cromossomo[gene] = random.randint(*self.bound_size[gene])
                    case "INT-PERM":
                        g2 = random.randint(0, len(cromossomo) - 1)
                        while g2 == gene:
                            g2 = random.randint(0, len(cromossomo) - 1)
                        aux = cromossomo[gene]
                        cromossomo[gene] = cromossomo[g2]
                        cromossomo[g2] = aux
                    case "REAL":
                        # cr_std = cromossomo.std()
                        # print("cr_std", cr_std)
                        # cromossomo[gene] = alelo + random.random(-cr_std, cr_std)
                        cromossomo[gene] = (
                            random.random()
                            * (self.bound_size[gene][1] - self.bound_size[gene][0])
                            + self.bound_size[gene][0]
                        )
                    case "CUSTOM":
                        cromossomo[gene] = self.problem._mutate(cromossomo, gene)
                    case _:
                        return random.choices((1, 0), k=self.dim_size)
        return cromossomo

    def generate_mutation(self, population):
        mutated_population = [self._mutate(cromossomo) for cromossomo in population]
        return mutated_population

    def _one_point_cross_over(self, cr1, cr2):
        position = random.randint(0, len(cr1) - 1)
        cr1_floor, cr1_ceil = cr1[:position], cr1[position:]
        cr2_floor, cr2_ceil = cr2[:position], cr2[position:]
        mated_cr1 = np.concatenate((cr1_floor, cr2_ceil), axis=0)
        mated_cr2 = np.concatenate((cr2_floor, cr1_ceil), axis=0)
        # print('cr:',cr1,cr2)
        # print('op:',cr1_floor,cr1_ceil,cr2_floor,cr2_ceil)
        # print('mated',mated_cr1,mated_cr2)
        return mated_cr1, mated_cr2

    def _cycle_crossover(self, cr1, cr2):
        first = cr1[0]
        gene1 = cr1[0]
        gene2 = cr2[0]
        max = len(cr1)
        sections_aux = np.array([False for _ in range(max)])
        i = 0
        while gene2 != first:
            i = (i + 1) % max
            gene1 = cr1[i]
            if gene1 == gene2:
                #sections.append(i)
                sections_aux[i] = True
                gene2 = cr2[i]
        mated_cr1 = np.array([cr1[i] if sections_aux[i] is True else cr2[i] for i in range(max)])
        mated_cr2 = np.array([cr2[i] if sections_aux[i] is True else cr1[i] for i in range(max)])
        return mated_cr1, mated_cr2

    def generate_cross_over(self, population):
        mated_population = []
        for i in range(len(population) // 2):
            cr1 = population[2 * i]
            cr2 = population[(2 * i) + 1]
            chance = random.random()
            if chance <= self.cross_over_rate:
                if self.config["COD"] == "INT-PERM":
                    mated_cr1, mated_cr2 = self._cycle_crossover(cr1, cr2)
                else:
                    mated_cr1, mated_cr2 = self._one_point_cross_over(cr1, cr2)
                mated_population += [mated_cr1, mated_cr2]
            else:
                mated_population += [cr1, cr2]
        return mated_population
    
    def dizimate_population(self, population):
        roman_line = random.sample(range(self.pop_size), self.pop_size//2)
        for soldier in roman_line:
            population[soldier] = self.gerar_individuo(soldier)
        return population


    def loop(self):
        self.save_elite()
        mating_pool = self.generate_mating_pool()
        intermediary_population = self.generate_cross_over(mating_pool)
        intermediary_population = self.generate_mutation(intermediary_population)
        intermediary_evaluation = self.evaluate(intermediary_population)

        if len(self.results_best) > 1:
            if self.results_best[-1] == self.results_best[-2]:
                self.counter +=1
            else:
                self.counter = 0
                
        if self.counter >= self.dizimation_interval:
            intermediary_population = self.dizimate_population(intermediary_population)
            intermediary_evaluation = self.evaluate(intermediary_population)
            self.counter = 0

        for elite, elite_eval in zip(self.elite_population, self.elite_evaluation):
            pior = intermediary_evaluation.argmin()
            intermediary_population[pior] = elite
            intermediary_evaluation[pior] = elite_eval
        self.population = intermediary_population
        self.evaluation = intermediary_evaluation

        # Add Results
        self.results_best.append(self.evaluation.max())
        self.results_mean.append(self.evaluation.mean())
        if self.save_penality is True:
            self.results_penality.append(
                self.problem.penality_function(
                    self.problem.decode(
                        self.population[self.evaluation.argmax()]
                    )
                )
            )




    def run(self, step=10,show=False):
        if show is True:
            print("Execution started:") 
        for _ in tqdm(range(self.iterations)):
            self.loop()
        self.save_elite()
        best = self.problem.decode(self.elite_population[0])
        if show is True:
            print("Best Solution: ", best)
            print("Fitness:", self.elite_evaluation[0])
            print("Objective value:", self.problem.objective_function(best))
