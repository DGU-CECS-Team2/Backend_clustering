from sklearn.manifold import TSNE

class TSNEReducer:
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def reduce(self, data):
        # 샘플 수에 따라 perplexity 값을 조정
        perplexity_value = min(30, len(data) - 1)
        tsne = TSNE(n_components=self.n_components, random_state=self.random_state, perplexity=perplexity_value)
        return tsne.fit_transform(data)


