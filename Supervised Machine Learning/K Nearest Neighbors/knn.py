from util import *

class KNN ():
    def __init__(self, k):
        self.k = k
    
    def k_nearest_neighbors(point, 
                        training_features, 
                        training_labels, 
                        k):
        # Create an empty list to store neighbors and distances
        neighbors = []
        
        for p, label in zip(training_features, training_labels):
            d = distance(point, p)
            temp_data = [p, label, d]
            neighbors.append(temp_data)
            
        neighbors.sort(key = lambda x : x[-1])
        
        return neighbors[:k]