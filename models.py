import numpy as np

class KMeans:
    def __init__(self, K, max_iters=100, tol=1e-4):
        '''
        Inicializa a classe KMeans.

        # Entradas:
        int: Número de clusters.
        int: Número máximo de iterações.
        float: Tolerância para verificar a convergência.
        '''
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.cluster_assignments = None
        self.final_loss = None
        self.norm = None
    
    def normalize_data(self, X):
        '''
        Normaliza os dados usando a padronização Min-Max para ficar no intervalo [0, 1].
    
        # Entrada:
        numpy.ndarray: Matriz dos dados.
    
        # Saída:
        numpy.ndarray: Matriz dos dados normalizados.
        '''
        dados_normalizados = []
        for data_point in X.T:  # Trabalhando com variáveis como linhas
            min_val = np.min(data_point)
            max_val = np.max(data_point)
            data_point_normalizado = [(valor - min_val) / (max_val - min_val) for valor in data_point]
            dados_normalizados.append(data_point_normalizado)

        return np.array(dados_normalizados).T  # Transpondo de volta

    def initialize_centroids(self, X):
        '''
        Inicializa os centróides selecionando K pontos aleatórios do conjunto de dados.

        # Entrada:
        numpy.ndarray: Matriz dos dados.

        # Saída:
        numpy.ndarray: Matriz dos centróides.
        '''
        indices = np.random.choice(X.shape[0], self.K, replace=False)
        centroids = X[indices]
        return centroids

    def assign_clusters(self, X, centroids):
        '''
        Atribui cada ponto de dados ao centroide mais próximo.

        # Entradas:
        numpy.ndarray: Matriz de dados X.
        numpy.ndarray: Matriz de centróides.

        # Saída:
        numpy.ndarray: Array contendo o índice do cluster mais próximo para cada ponto.
        '''
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def update_centroids(self, X, cluster_assignments):
        '''
        Atualiza os centróides calculando a média dos pontos atribuídos a cada centroide.

        # Entradas:
        numpy.ndarray: Matriz de dados X.
        numpy.ndarray: Array contendo o índice do cluster para cada ponto.

        # Saída:
        numpy.ndarray: Matriz de centróides atualizados.
        '''
        centroids = np.array([X[cluster_assignments == k].mean(axis=0) for k in range(self.K)])
        return centroids

    def compute_loss(self, X, centroids, cluster_assignments):
        '''
        Calcula a distorção (erro de reconstrução) no algoritmo K-means.

        # Entradas:
        numpy.ndarray: Matriz de dados X.
        numpy.ndarray: Matriz de centróides.
        numpy.ndarray: Array contendo o índice do cluster para cada ponto.

        # Saída:
        float: Valor da distorção.
        '''
        loss = np.sum((X - centroids[cluster_assignments])**2)
        return loss

    def fit(self, X):
        '''
        Realiza o treinamento do algoritmo K-means.

        # Entrada:
        numpy.ndarray: Matriz de dados X.

        # Saída:
        None
        '''
        
        X = self.normalize_data(X)  # Normaliza os dados
        print('Dados não normalizados')
            
        self.centroids = self.initialize_centroids(X)
        
        for i in range(self.max_iters):
            self.cluster_assignments = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, self.cluster_assignments)
            
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            
            self.centroids = new_centroids
        
        self.final_loss = self.compute_loss(X, self.centroids, self.cluster_assignments)

    def predict(self, X):
        '''
        Realiza a previsão dos clusters para novos dados.

        # Entrada:
        numpy.ndarray: Matriz de dados X.

        # Saída:
        numpy.ndarray: Array contendo o índice do cluster para cada ponto.
        '''
        X = self.normalize_data(X)  # Normaliza os dados
        return self.assign_clusters(X, self.centroids)

    def get_centroids(self):
        '''
        Retorna os centróides finais após o treinamento.

        # Saída:
        numpy.ndarray: Matriz de centróides.
        '''
        return self.centroids

    def get_loss(self):
        '''
        Retorna a distorção final após o treinamento.

        # Saída:
        float: Valor da distorção.
        '''
        return self.final_loss
