import random
import numpy as np

# Define the population size and mutation rate
POP_SIZE = 50
MUTATION_RATE = 0.1

# Define the neural network architecture
input_size = 10
hidden_size = 5
output_size = 1

# Create a random population of weight sets


def create_population():
    population = []
    for i in range(POP_SIZE):
        # Generate random weights for each layer
        w1 = np.random.randn(input_size, hidden_size)
        w2 = np.random.randn(hidden_size, output_size)
        population.append((w1, w2))
    return population

# Define the fitness function


def fitness_function(weights):
    # Train the neural network with the given weights
    # and evaluate its performance on a validation set
    # Return a fitness score based on the accuracy or other metric
    # For simplicity, let's just return a random fitness score
    return random.uniform(0, 1)

# Select the fittest individuals from the population


def selection(population):
    # Choose the top n individuals with the highest fitness scores
    sorted_population = sorted(
        population, key=lambda x: fitness_function(x), reverse=True)
    return sorted_population[:int(0.2*POP_SIZE)]

# Create new individuals by combining the weights of parents


def crossover(parents):
    # Perform uniform crossover to create new weight sets
    w1s, w2s = zip(*parents)
    w1_new = np.zeros_like(w1s[0])
    w2_new = np.zeros_like(w2s[0])
    for i in range(w1_new.shape[0]):
        for j in range(w1_new.shape[1]):
            w1_new[i, j] = random.choice(w1s)[i, j]
    for i in range(w2_new.shape[0]):
        for j in range(w2_new.shape[1]):
            w2_new[i, j] = random.choice(w2s)[i, j]
    return (w1_new, w2_new)

# Introduce random mutations to the weights


def mutation(weights):
    # Add random noise to each weight
    w1, w2 = weights
    w1 += np.random.randn(*w1.shape) * MUTATION_RATE
    w2 += np.random.randn(*w2.shape) * MUTATION_RATE
    return (w1, w2)


# Run the genetic algorithm
population = create_population()
for generation in range(10):
    # Select the fittest individuals from the population
    parents = selection(population)

    # Create new individuals through crossover and mutation
    offspring = []
    for i in range(POP_SIZE - len(parents)):
        parent1, parent2 = random.choices(parents, k=2)
        child = crossover([parent1, parent2])
        child = mutation(child)
        offspring.append(child)

    # Replace the old population with the new population
    population = parents + offspring

    # Evaluate the fitness of the new population
    fitnesses = [fitness_function(individual) for individual in population]

    # Print some statistics for this generation
    print("Generation", generation+1)
    print("  Best fitness:", max(fitnesses))
    print("  Average fitness:", sum(fitnesses) / len(fitnesses))
