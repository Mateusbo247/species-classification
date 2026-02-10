import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random
import os


# Função para carregar dados
def load_data(file_path):
    """
    Carrega os dados do arquivo CSV e separa em ID, rótulos reais e previsões de ambos os modelos.
    """
    data = pd.read_csv(file_path)
    data = data.dropna()
    ids = data.iloc[:, 0].values  # ID dos registros
    y_true = data.iloc[:, 16].values  # Saídas reais
    model_1_probs = data.iloc[:, 1:16].values  # Probabilidades do Modelo 1
    model_2_probs = data.iloc[:, 17:32].values  # Probabilidades do Modelo 2
    return ids, y_true, model_1_probs, model_2_probs

# Função de avaliação
def evaluate(y_true, combined_probs):
    """
    Avalia a precisão combinando as previsões e calculando a acurácia.
    """
    y_pred = np.argmax(combined_probs, axis=1)
    return accuracy_score(y_true, y_pred)

# Combinação das probabilidades dos modelos
def combine_probabilities(model_1_probs, model_2_probs, weights):
    """
    Combina as probabilidades dos dois modelos usando pesos específicos para cada classe.
    """
    return weights * model_1_probs + (1 - weights) * model_2_probs

# Função para criar a população inicial
def create_population(size, num_classes):
    """
    Cria uma população inicial de pesos aleatórios para cada classe.
    """
    return np.random.rand(size, num_classes)

# Seleção dos melhores indivíduos
def select(population, scores, num_parents):
    """
    Seleciona os melhores indivíduos da população com base na pontuação.
    """
    parents = []
    pop_size = len(population)
    
    for _ in range(num_parents):
        id1, id2 = np.random.randint(0, pop_size, size = 2)
        if scores[id1] >= scores[id2]:
            parents.append(population[id1])
        else:
            parents.append(population[id2])
    return np.array(parents)
    
# Seleção dos melhores indivíduos
def reduce_pop(population, scores, num_parents):
    """
    Seleciona os melhores indivíduos da população com base na pontuação.
    """
    selected_indices = np.argsort(scores)[-num_parents:]
    return population[selected_indices]

# Função de cruzamento
def crossover(parents, offspring_size, num_classes):
    """
    Realiza o cruzamento entre os pais para gerar novos indivíduos.
    """
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(list(parents), 2)
        child = (parent1 + parent2) / 2
        offspring.append(child)
    return np.array(offspring)

# Função de mutação
def mutate(offspring, mutation_rate=0.1):
    """
    Aplica mutação nos indivíduos da população.
    """
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            mutation = np.random.normal(0, 0.1, offspring.shape[1])
            offspring[i] += mutation  # Pequena alteração
            offspring[i] = np.clip(offspring[i], 0, 1)  # Mantém os pesos entre 0 e 1
    return offspring

# Algoritmo genético principal
def genetic_algorithm(y_true, model_1_probs, model_2_probs, y_true_t, model_1_probs_t, model_2_probs_t, num_generations=60, population_size=100, num_parents=100, mutation_rate=0.5):
    """
    Executa o algoritmo genético para encontrar os melhores pesos por classe.
    """
    num_classes = model_1_probs.shape[1]
    population = create_population(population_size, num_classes)  # População inicial
    scores = []
    for weights in population:
        combined_probs = combine_probabilities(model_1_probs, model_2_probs, weights)
        accuracy = evaluate(y_true, combined_probs)
        scores.append(accuracy)
    scores = np.array(scores)
    acc_plot = []
    acc_plot_t = []
    for generation in range(num_generations):
        
        parents = select(population, scores, num_parents)
        
        # Cruzamento e mutação para gerar nova população
        offspring_size = num_parents
        offspring = crossover(parents, offspring_size, num_classes)
        offspring = mutate(offspring, mutation_rate)
        
        population = reduce_pop(population, scores, 5)
        
        scores_off = []
        for weights in offspring:
            combined_probs = combine_probabilities(model_1_probs, model_2_probs, weights)
            accuracy = evaluate(y_true, combined_probs)
            scores_off.append(accuracy)
        scores_off = np.array(scores_off)
       
        offspring = reduce_pop(offspring, scores_off, population_size - 5)
                
        # Atualiza a população
        population = np.concatenate((population, offspring))
        
        scores = []
        for weights in population:
            combined_probs = combine_probabilities(model_1_probs, model_2_probs, weights)
            accuracy = evaluate(y_true, combined_probs)
            scores.append(accuracy)
        
        # Seleção dos melhores
        scores = np.array(scores)
        
        # Melhor indivíduo desta geração
        if (generation == 0):
            best_score = max(scores)
            best_ind = np.argmax(scores)
            best_weights = population[best_ind]
        elif (max(scores)> best_score):
            best_score = max(scores)
            best_ind = np.argmax(scores)
            best_weights = population[best_ind]
        print(f"Geração {generation + 1} - Melhor precisão: {best_score:.4f}")
        acc_plot.append(best_score)
        combined_probs_t = combine_probabilities(model_1_probs_t, model_2_probs_t, best_weights)
        accuracy_t = evaluate(y_true_t, combined_probs_t)
        acc_plot_t.append(accuracy_t)
    
    # Melhor conjunto de pesos encontrado
    print(f"Geração {generation + 1} - Melhor precisão: {best_score:.4f}")
    return best_weights, np.array(acc_plot), np.array(acc_plot_t)

pastas = [p for p in os.listdir('/home/antonio/mateus_results/predictions_for_ga/') if os.path.isdir(os.path.join('/home/antonio/mateus_results/predictions_for_ga/', p))]

for model in pastas:
    acc_plots = None
    acc_plots_t = None
    df_results = pd.DataFrame(index=range(42, 72), columns=range(1, 2))
    df_val = pd.DataFrame(index=range(42, 72), columns=range(1, 2))
    indiv = []
    for SEED in range(42,72):
        # Python built-in
        random.seed(SEED)
        # NumPy
        np.random.seed(SEED)
        print(f' Iniciando SEED {SEED} para modelo {model}')
        # Carregar os dados
        file_path = f'/home/antonio/mateus_results/predictions_for_ga/{model}/val_predictions.csv'  # Substitua pelo caminho do seu arquivo
        train_path = f'/home/antonio/mateus_results/predictions_for_ga/{model}/train_predictions.csv'
        ids, y_true, model_1_probs, model_2_probs = load_data(file_path)
        ids_t, y_true_t, model_1_probs_t, model_2_probs_t = load_data(train_path)
        # Executar o algoritmo genético
        melhores_pesos, acc_plot, acc_plot_t = genetic_algorithm(y_true, model_1_probs, model_2_probs, y_true_t, model_1_probs_t, model_2_probs_t)
        print(f"Melhores pesos encontrados para cada classe: {melhores_pesos}")
        combined_probs = combine_probabilities(model_1_probs, model_2_probs, melhores_pesos)
        accuracy = evaluate(y_true, combined_probs)
        df_val.at[SEED, 1] = round(100 * accuracy, 2)
        indiv.append(melhores_pesos)
        if acc_plots is None:
            acc_plots = acc_plot.copy()
        else:
            acc_plots += acc_plot
        if acc_plots_t is None:
            acc_plots_t = acc_plot_t.copy()
        else:
            acc_plots_t += acc_plot_t
        ###########################################################################################################
        # Carregar os dados
        file_path = f'/home/antonio/mateus_results/predictions_for_ga/{model}/test_predictions.csv'  # Substitua pelo caminho do seu arquivo
        ids, y_true, model_1_probs, model_2_probs = load_data(file_path)
        combined_probs = combine_probabilities(model_1_probs, model_2_probs, melhores_pesos)
        accuracy = evaluate(y_true, combined_probs)
        print(f"\nPrecisão de TESTE: {accuracy:.4f}\n")
        df_results.at[SEED, 1] = round(100 * accuracy, 2)
    acc_plots = acc_plots/30
    acc_plots_t = acc_plots_t/30
    #np.savetxt(f'/home/antonio/mateus_results/ga_outputs/15_weights_classic_tournament/convergencia_treino/{model}.csv', acc_plots, delimiter = ";", header = "acc_mean")
    #np.savetxt(f'/home/antonio/mateus_results/ga_outputs/15_weights_classic_tournament/convergencia_treino/{model}_treino.csv', acc_plots_t, delimiter = ";", header = "acc_mean")
    df_results.to_csv(f'/home/antonio/mateus_results/ga_outputs/15_weights_classic_tournament/test_results_{model}.csv', index_label="RUN")
    df_val.to_csv(f'/home/antonio/mateus_results/ga_outputs/15_weights_classic_tournament/val_{model}.csv', index_label="RUN")
    df_pesos = pd.DataFrame(indiv, columns = ["anfibio", "ave-de-rapina", "cachorro", "capivara", "cobra", "cutia", "gamba", "lagarto", "macaco", "mico", "morcego", "preguica", "quati", "tamandua", "tartaruga"])
    df_pesos.to_csv(f'/home/antonio/mateus_results/ga_outputs/15_weights_classic_tournament/PESOS_{model}.csv', index=False)

