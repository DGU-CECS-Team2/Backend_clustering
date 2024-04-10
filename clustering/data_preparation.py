import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreparer:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def summarize_interests(self, data):
        interests = np.mean(data[:, 4:7], axis=1, keepdims=True)
        summarized_data = np.hstack((data[:, :4], interests))
        return summarized_data

    def scale_data(self, data):
        return self.scaler.fit_transform(data)
