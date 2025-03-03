import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import struct
import pdb

# *************************************** Rede Bioinspirada ******************************************

def gennet_inh_lat(input_neuron, compete_neuron, autapse):
  m11 = np.zeros((input_neuron, input_neuron))
  m12 = np.zeros((input_neuron, compete_neuron))

  m21 = 0.1 * np.ones((compete_neuron, input_neuron)) + 0.1 * np.random.rand(compete_neuron, input_neuron)
  m22 = -(0.8 * (np.ones((compete_neuron, compete_neuron)) - np.eye(compete_neuron))) + autapse * np.eye(compete_neuron)

  w = np.block([[m11, m12], [m21, m22]])

  m22 = np.zeros_like(m22)
  w_mask = np.block([[m11, m12], [m21, m22]])
  mask = w_mask > np.zeros_like(w_mask)

  return w, mask


def performance_analysis(output_graf, number_patterns):
  try:
    limiar = 0.9
    df = pd.DataFrame(output_graf).tail(number_patterns)
    df["Sobreposição"] = df[[5, 6, 7]].apply(lambda row: (row > limiar).sum(), axis=1)
    overlap_summary = df["Sobreposição"].value_counts().sort_index()

    non_overlap_percentage = overlap_summary.loc[1] / overlap_summary.sum() # total linhas com 1 sobreposicao / total linhas

    return non_overlap_percentage

  except Exception as e:
    return 0
  
def bioinspired_neural_network(autapse, learning_factor, displacement_speed, is_test):
    P = np.array([
        [0.1, 0.2, 0.0, 0.0, 0.7],  # A
        [0.0, 0.0, 0.4, 0.6, 0.0],  # B
        [0.3, 0.1, 0.0, 0.0, 0.6],  # A
        [0.8, 0.2, 0.0, 0.0, 0.0],  # C
        [0.0, 0.0, 0.5, 0.5, 0.0],  # B
        [0.0, 0.0, 0.6, 0.4, 0.0],  # B
        [0.2, 0.1, 0.0, 0.0, 0.7],  # A
        [0.9, 0.1, 0.0, 0.0, 0.0],  # C
        [0.7, 0.3, 0.0, 0.0, 0.0],  # C
    ]).T
    
    n_columns, n_lines = P.shape
    w, mask = gennet_inh_lat(n_columns, 3, autapse)
    n_neurons = w.shape[0]
    
    # Parâmetros
    shift = 0.5 * np.ones((n_neurons, 1))
    learning_factor = learning_factor
    displacement_speed = displacement_speed
    epochs = 110
    
    # Inicializações
    incw = np.zeros_like(w)
    output_before = np.zeros((n_neurons, 1))
    output = np.zeros_like(output_before)
    n_entries, standards = P.shape
    layers = 1
    inter_totals = 1
    
    output_graf = []
    
    # Treinamento
    for i in range(epochs):
        for j in range(standards):
            output = np.zeros((n_neurons, 1))
            output_before = output.copy()
            
            PAT = P[:, j].reshape(-1, 1)
            output[:n_entries, 0] = PAT.flatten()
            
            for k in range(layers + 1):
                w += incw
                Inet = w @ output  # Multiplicação matricial
                output = 1 / (1 + np.exp(-59 * (Inet - shift)))
                output = (Inet > 0) * output  # Aplica limiar de ativação
                output[:n_entries, 0] = PAT.flatten()
                
                incw = (learning_factor * (output @ output_before.T - (1 + 0.05) * np.ones_like(output) @ output_before.T * w)) * mask
                shift = (displacement_speed * output + shift) / (1 + displacement_speed)
                output_before = output.copy()
            
            if i >= (epochs - standards):
                output_graf.append(output.flatten())
                inter_totals += 1
    
    output_graf = np.array(output_graf)

    if is_test == True:
        return output_graf[-n_lines:]
        
    return performance_analysis(output_graf, n_lines)


# *************************************** Algoritmo genético ******************************************

# Função que gera um cromossomo com três parâmetros
def generate_chromosome():
    return {
        "param1": round(random.uniform(0, 1.0), 6), #autapse
        "param2": round(random.uniform(0, 0.01), 6), #fator aprendizado
        "param3": round(random.uniform(0, 0.01), 6), #fator deslocamento
        "fitness": 0
    }

# Função de avaliação
def evaluate_function(chromo):
    non_overlap_percentage = bioinspired_neural_network(chromo["param1"], chromo["param2"], chromo["param3"], False)
    chromo["fitness"] = non_overlap_percentage
    return chromo

# Seleção via roleta
def roulette_selection(population):
    total_fitness = sum(c["fitness"] for c in population)
    pick = random.uniform(0, total_fitness)
    cumulative = 0
    for chromo in population:
        cumulative += chromo["fitness"]
        if cumulative >= pick:
            return chromo
    return population[0]

# Função de crossover uniforme
def crossover_uniform(parent1, parent2):
    return {
        "param1": parent1["param1"] if random.random() < 0.5 else parent2["param1"],
        "param2": parent1["param2"] if random.random() < 0.5 else parent2["param2"],
        "param3": parent1["param3"] if random.random() < 0.5 else parent2["param3"],
        "fitness": parent1["fitness"] if random.random() < 0.5 else parent2["fitness"]
    }

# Converter float para bit
def float_to_bits(f):
    # Converte um float para sua representação em bits (inteiro)
    return struct.unpack('!I', struct.pack('!f', f))[0]

# Converter bit para float
def bits_to_float(b):
    # Converte um inteiro (representação em bits) de volta para float
    return struct.unpack('!f', struct.pack('!I', b))[0]

# Mutação
def mutate(chromosome):
    bit_to_flip = random.randint(0, 23)  # Ponto de corte
    mask = 1 << (bit_to_flip % 8)
    component = ["param1", "param2", "param3"][bit_to_flip // 8]
    
    # Converte o valor float para bits, aplica a mutação e converte de volta
    bits = float_to_bits(chromosome[component])
    bits ^= mask
    chromosome[component] = bits_to_float(bits)
    
    return chromosome

# Algoritmo Genético principal
def main_genetic_algorithm():
    population = [generate_chromosome() for _ in range(10)]
    best_chromosomes_in_round = []
    max_iterations = 120
    iteration = 0
    chromo_list = []

    print("\nExecutando o Algoritmo Genético...")

    while iteration <= max_iterations:        
        for i, chromo in enumerate(population):
            chromo = evaluate_function(chromo)
            chromo_list.append(chromo)
            
        for chromo in chromo_list:
            if chromo["fitness"] >= 0.1:
                best_chromosomes_in_round.append(chromo)

        # Se for a ultima interacao, pego o ultimo resultado
        if iteration == max_iterations:
            population = best_chromosomes_in_round
            break

        if len(best_chromosomes_in_round) >= 2:
            
            # Evolução da população
            new_population = []
            for _ in range(4):
                parent1 = roulette_selection(best_chromosomes_in_round)
                parent2 = roulette_selection(best_chromosomes_in_round)
     
                offspring = crossover_uniform(parent1, parent2)
                offspring_mutated = mutate(offspring)
                new_population.append(offspring_mutated)
    
            for _ in range(6):
                new_population.append(generate_chromosome())
            
            population.clear()
            chromo_list.clear()
            best_chromosomes_in_round.clear()
            population = new_population
    
        iteration += 1

    if len(population) == 0:
        return population
    else:
        return max(population, key=lambda x: x['fitness'])

def genetic_optimizer():
    solution = main_genetic_algorithm()
    validador = True

    while validador == True:
        if len(solution) == 0 :
            solution = main_genetic_algorithm()
            validador = True

        else:
            if solution['fitness'] == 1.0 or solution['fitness'] is not None:
                print(f"\nMelhor solução encontrada: {solution}\n")
                validador = False

            else:
                solution = main_genetic_algorithm()
                validador = True

    return solution            

# Função para identificar qual neurônio acertou
def identify_hit_neuron(row):
    if row[5] >= 0.7:
        return '5'
    elif row[6] >= 0.7:
        return '6'
    elif row[7] >= 0.7:
        return '7'
    else:
        return None

def main():
  # Executar o otimizador genético
  result = genetic_optimizer()

  autapse = result['param1']
  learning_factor = result['param2']
  displacement_speed = result['param3']

  standards = bioinspired_neural_network(autapse, learning_factor, displacement_speed, True)
  df = pd.DataFrame(standards)

  df['Padrão'] = ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C', 'C']
  df['Neurônio Acerto'] = df.apply(identify_hit_neuron, axis=1)
  
  print(df)

  desempenho = performance_analysis(standards, len(standards))
  print(f"\nDesempenho: {desempenho}\n")
  
  selected_columns = df[[5, 6, 7]]

  # Criar um gráfico de linha
  plt.figure(figsize=(8, 6))  # Define o tamanho da figura
  for col in selected_columns.columns:
      plt.plot(selected_columns.index, selected_columns[col], label=f'Neuronio {col}')

  # Configurar o gráfico
  plt.title("Disparo dos neuronios 5, 6 e 7")
  plt.xlabel("Tempo")
  plt.ylabel("Grau de disparo")
  plt.legend(title="Neurônios")
  plt.grid(True)

  # Exibir o gráfico
  plt.show()

if __name__ == '__main__':
  print("Inicializando o programa")
  main()


