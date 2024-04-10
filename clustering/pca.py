from sklearn.decomposition import PCA

class PCAReducer:
    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)
    
    def reduce(self, data):
        return self.pca.fit_transform(data)
