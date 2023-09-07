import random
import numpy as np
from types import NoneType
from joblib import Parallel,delayed


class Problem:
    def encode(self, solution) -> np.array:
        ...

    def decode(self, cromossomo: np.array) -> any:
        ...

    def objective_function(self, solution) -> any:
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

    def __init__(self, config: dict, problem: Problem,parallel=False) -> None:
        random.seed()
        self.parallel = parallel
        self.problem = problem
        self.config = config
        self.config = problem.set_problem(config)
        print('bound:',self.config['BOUND'])
        self.pop_size = int(self.config["POP"])
        self.dim_size = int(self.config["DIM"])
        self.bound_size = eval(self.config['BOUND'])
        self.population = self.generate_population()
        self.evaluation = self.evaluate(self.population)

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
            int(self.config['ITERATIONS'])
            if 'ITERATIONS' in self.config.keys()
            else 100
        )
        self.elite_population = []
        self.elite_evaluation = []
        self.mating_pool = []

        self.results_best = []
        self.results_mean = []

        self.mutation_rate = 0.05

    def generate_population(self):
        if self.config["COD"] == "CUSTOM-INT":
            return self.problem.generate_population(self.pop_size)
        else:
            population = [self.gerar_individuo(gene) for gene in range(self.pop_size)]
            return population

    def gerar_individuo(self,gene:int):
        dim = int(self.config["DIM"])
        match self.config["COD"]:
            case "BIN":
                individuo = np.array(random.choices((1, 0), k=dim))
                return individuo
            case "INT":
                #bound = list(map(int, self.config["BOUND"].strip("][ ").split(",")))
                #bound = eval(self.config['BOUND'])
                individuo = np.array([random.randint(*self.bound_size[gene]) for _ in range(dim)])
                return individuo
            case "INT-PERM":
                #bound = (0, dim)
                individuo = random.sample(range(*self.bound_size[gene]), k=dim)
                assert len(set(individuo)) == len(individuo)
                return np.array(individuo)
            case "REAL":
                #bound = list(map(int, self.config["BOUND"].strip("][ ").split(",")))
                individuo = np.array(
                    [
                        random.random() * (self.bound_size[gene][1] - self.bound_size[gene][0]) + self.bound_size[gene][0]
                        for _ in range(dim)
                    ]
                )
                return individuo
            case _:
                return random.choices((1, 0), k=dim)

    def evaluate(self, population=None) -> np.ndarray:
        """Avalia uma população, caso não seja fornecido uma população como argumento, avalia self.population e atualiza/retorna self.evaluation"""
        if type(population) == NoneType:
            evaluation = np.array(
                [
                    self.problem.fitness(self.problem.decode(cromossomo))
                    for cromossomo in self.population
                ]
            )
            def fx(x):
                print(x)
                self.problem.fitness(self.problem.decode(x))
                return x
            evaluation = np.array(Parallel(n_jobs=4)
                (
                    delayed(fx)(cromossomo)
                    for cromossomo in self.population
                )
            )
            return evaluation
        else:
            if self.parallel is False:
                evaluation = np.array(
                    [
                        self.problem.fitness(self.problem.decode(cromossomo))
                        for cromossomo in self.population
                    ]
                )
            else:
                # def fx(x):
                #     return self.problem.fitness()
                evaluation = np.array(Parallel(n_jobs=-1)
                    (
                        delayed(self.problem.fitness)(self.problem.decode(cromossomo))
                        for cromossomo in self.population
                    )
                )
            return evaluation

    def save_elite(self):
        #print('eval:',self.evaluation)
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

    def generate_mating_pool(self):
        """Gera a Mating pool com base na população"""
        mating_pool = np.array(
            [
                self.roulette_wheel()[i]
                for i in range(0, 2)
                for _ in range(self.pop_size // 2)
            ]
        )
        self.mating_pool = mating_pool
        return mating_pool

    def _mutate(self, cromossomo: np.ndarray):
        chance = random.random()   
        if chance <= self.mutation_rate:
            gene = random.randint(0,len(cromossomo)-1)
            alelo = cromossomo[gene]
            match self.config["COD"]:
                case "BIN":
                    cromossomo[gene] = not alelo
                case "INT":
                    # cr_std = cromossomo.std()
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
                    cromossomo[gene] = random.random() * (self.bound_size[gene][1] - self.bound_size[gene][0]) + self.bound_size[gene][0]
                case "CUSTOM":
                    cromossomo[gene] = self.problem._mutate(cromossomo,gene)
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

    # def _partially_matched_crossover(self,cr1,cr2):
    #     half = len(cr1)//2
    #     p1 = random.randint(0,len(cr1)-1)
    #     p2
    def _cycle_crossover(self, cr1, cr2):
        sections = [0]
        gene2 = cr2[0]
        first = cr1[0]
        max = len(cr1)
        i = 0
        while gene2 != first:
            i = (i + 1) % max
            gene1 = cr1[i]
            if gene1 == gene2:
                sections.append(i)
                gene2 = cr2[i]

        mated_cr1 = np.array([cr1[i] if i in sections else cr2[i] for i in range(max)])
        mated_cr2 = np.array([cr2[i] if i in sections else cr1[i] for i in range(max)])
        # print('cr1:',cr1)
        # print('cr2:',cr2)
        # print('mated cr1:',mated_cr1)
        # print('mated cr2:',mated_cr2)
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

    def loop(self):
        self.save_elite()
        # print('Pop:',self.population)
        self.generate_mating_pool()
        self.mating_pool = self.generate_cross_over(self.mating_pool)
        self.mating_pool = self.generate_mutation(self.mating_pool)
        # print('Mutated:',self.mating_pool)
        mating_pool_evaluation = self.evaluate(self.mating_pool)
        # print([self.problem.decode(cromossomo) for cromossomo in self.mating_pool])
        for elite, elite_eval in zip(self.elite_population, self.elite_evaluation):
            pior = mating_pool_evaluation.argmin()
            self.mating_pool[pior] = elite
            mating_pool_evaluation[pior] = elite_eval
        self.population = self.mating_pool
        self.evaluation = mating_pool_evaluation

        # Add Results
        self.results_best.append(self.evaluation.max())
        self.results_mean.append(self.evaluation.mean())

        # print('Elite eval:',self.elite_evaluation)
        # end of execution

    def run(self,step=10):
        print('Execution started:')
        for i in range(self.iterations):
            if i % step ==0:
                print('{:.2f}'.format((i / self.iterations)*100),'%')
            self.loop()
        self.save_elite()
        best = self.problem.decode(self.elite_population[0])
        print('Best Solution: ',best)
        print('Fitness:',self.elite_evaluation[0])
        print('Objective value:',self.problem.objective_function(best))