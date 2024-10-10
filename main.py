import torch
import pandas as pd
import random
import logging
from GA import GeneticAlgorithm
from env import RouteAssignmentEnv

# Set random seed
random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import data
df = pd.read_excel('preprocessed.xlsx')
df['ActualArrival'] = pd.to_datetime(df['ActualArrival'], format="%H:%M:%S")

busNo = df['ServiceNo'].unique()
busNo.sort()
bus_data = pd.DataFrame({'ServiceNo': busNo})

# Set parameters
alpha = 0.7     # Ratio of berth1' delay
beta = 0.3      # Ratio of berth2' delay
theta = 10       # Ratio of distribution

#training
env = RouteAssignmentEnv(bus_data)
ga = GeneticAlgorithm(env, pop_size=50, generations=20, mutation_rate=0.1, df=df)
berth_1, berth_2, best_fitness = ga.run(alpha, beta, theta)

print(f"Best Solution - Berth 1: {berth_1}")
print(f"Best Solution - Berth 2: {berth_2}")
print(f"Best Fitness: {best_fitness}")

'''
Save the results
'''