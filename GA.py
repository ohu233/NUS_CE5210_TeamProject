import torch
import logging
import time

class GeneticAlgorithm:
    '''
    env: environment -> Env_GA.py
    pop_size: population per generation
    generations: epochs of training
    mutation_rate: ratio of mutation
    df: DataFrame -> preprocessed.xlsx
    elitism -> saving elite
    '''
    def __init__(self, env, pop_size, generations, mutation_rate, df, elitism=True):
        self.env = env
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
        self.population = self._initialize_population()
        self.df = df

    def _initialize_population(self):
        '''
        0 represents berth_1
        1 represents berth_2
        '''
        population = [torch.randint(0, 2, (self.env.num_buses,)).to(self.device).tolist() for _ in range(self.pop_size)]
        return population

    def fitness(self, individual, alpha, beta, theta, df):
        '''
        calculate fitness per individual
        '''
        berth_1, berth_2 = self.decode_individual(individual)

        reward = self.env.calculate_reward(alpha, beta, theta, berth_1, berth_2, df)
        return reward


    def ranking_selection(self, fitness_scores):
        '''
        other selection such as tournament selcetion can be chosen
        '''
        sorted_indices = torch.argsort(torch.tensor(fitness_scores))
        selection_probs = torch.tensor([len(fitness_scores) - i for i in range(len(fitness_scores))], dtype=torch.float32).to(self.device)  # Convert to float
        selection_probs /= torch.sum(selection_probs)  # Ensure proper division with float
        selected_idx = torch.multinomial(selection_probs, 1).item()
        return self.population[sorted_indices[selected_idx]]

    def crossover(self, parent1, parent2):
        '''
        Two-Point Crossover
        other corssover can be chosen
        '''
        point1 = torch.randint(1, self.env.num_buses - 1, (1,)).item()
        point2 = torch.randint(point1 + 1, self.env.num_buses, (1,)).item()
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2

    def mutate(self, individual, generation, stagnant_generations):
        '''
        dynamic mutation
        parameter could be changed
        '''
        current_mutation_rate = self.mutation_rate * (1 - generation / self.generations)
        if stagnant_generations > 3:
            current_mutation_rate = min(0.5, current_mutation_rate * 1.5)

        for i in range(len(individual)):
            if torch.rand(1).item() < current_mutation_rate:
                individual[i] = 1 - individual[i]

    def simulated_annealing(self, individual, alpha, beta, theta):
        '''
        simulated annealing
        parameter could be changed
        '''
        T = 500
        cooling_rate = 0.95
        current_solution = torch.tensor(individual, device=self.device)
        best_solution = current_solution.clone()
        best_fitness = self.fitness(best_solution.tolist(), alpha, beta, theta, self.df)

        iteration = 0
        max_iterations = 1000

        while T > 1 and iteration < max_iterations:
            num_changes = torch.randint(1, 3, (1,)).item()
            new_solution = current_solution.clone()

            for _ in range(num_changes):
                index = torch.randint(0, len(current_solution), (1,)).item()
                new_solution[index] = 1 - new_solution[index]

            new_fitness = self.fitness(new_solution.tolist(), alpha, beta, theta, self.df)  # `new_fitness` now is a float

            if new_fitness > best_fitness:
                best_solution = new_solution.clone()
                best_fitness = new_fitness
            else:
                delta_fitness = min((new_fitness - best_fitness) / T, 700)
                acceptance_prob = torch.exp(delta_fitness)
                if torch.rand(1, device=self.device).item() < acceptance_prob.item():
                    current_solution = new_solution.clone()

            T *= cooling_rate
            iteration += 1

        return best_solution.tolist()

    def decode_individual(self, individual):
        '''
        return list of route finally
        '''
        berth_1 = [self.env.bus_data.iloc[i]['ServiceNo'] for i in range(len(individual)) if individual[i] == 0]
        berth_2 = [self.env.bus_data.iloc[i]['ServiceNo'] for i in range(len(individual)) if individual[i] == 1]
        return berth_1, berth_2

    def run(self, alpha, beta, theta):
        best_individual = None
        best_fitness = float('-inf')
        stagnant_generations = 0

        for generation in range(self.generations):
            start_time = time.time()

            fitness_scores = [self.fitness(individual, alpha, beta, theta, self.df) for individual in self.population]

            current_best_fitness = max(fitness_scores)
            current_best_individual = self.population[torch.argmax(torch.tensor(fitness_scores)).item()]
            
            annealed_individual = self.simulated_annealing(current_best_individual, alpha, beta, theta)
            annealed_fitness = self.fitness(annealed_individual, alpha, beta, theta, self .df)
            
            if annealed_fitness > current_best_fitness:
                current_best_individual = annealed_individual
                current_best_fitness = annealed_fitness

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            for i, individual in enumerate(self.population):
                logging.info(f"Individual {i + 1}: {individual}, Fitness: {fitness_scores[i]}")

            logging.info(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
            logging.info(f"Best Individual: {best_individual}")

            new_population = [best_individual] if self.elitism else []

            while len(new_population) < self.pop_size:
                parent1 = self.ranking_selection(fitness_scores)
                parent2 = self.ranking_selection(fitness_scores)

                child1, child2 = self.crossover(parent1, parent2)

                self.mutate(child1, generation, stagnant_generations)
                self.mutate(child2, generation, stagnant_generations)

                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            self.population = new_population

            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"Generation {generation + 1} completed in {elapsed_time:.2f} seconds")

        berth_1, berth_2 = self.decode_individual(best_individual)
        return berth_1, berth_2, best_fitness