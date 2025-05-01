from util import *

class KNN ():
    def __init__(self, k, point, training_features, training_labels):
        self.k = k
        self.point = point
        self.training_features = training_features
        self.training_labels = training_labels
    
    def k_nearest_neighbors(self):
        # Create an empty list to store neighbors and distances
        neighbors = []
        
        for p, label in zip(self.training_features, self.training_labels):
            d = distance(self.point, p)
            temp_data = [p, label, d]
            neighbors.append(temp_data)
            
        neighbors.sort(key = lambda x : x[-1])
        
        return neighbors[:self.k]