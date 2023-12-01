import math
from sklearn.cluster import KMeans
import numpy as np

'''
def get_estimators(training_vector,clusters_number,segment_size,segments_number):
    estimators =[]
    training_vector = training_vector.reshape(segments_number, training_vector.shape[0], segment_size)
    for segment in training_vector:
        estimator = KMeans(n_clusters=clusters_number,random_state=42)
        estimator.fit(segment.reshape(-1,segment_size))
        estimators.append(estimator)
    return estimators

def get_predictions(original_vector,estimators,segment_size,segments_number):
    original_vector = original_vector.reshape(segments_number, segment_size)
    result_vector =[]
    for i,estimator in enumerate(estimators):
        segment_predictions = estimator.predict(original_vector[i].reshape(-1, segment_size))
        result_vector.append(segment_predictions[0])
    return result_vector

def PQ(original_vector,training_vector,segments_number,clusters_number):
    segment_size = math.ceil(len(original_vector)/segments_number)
    print(len(original_vector))
    estimators = get_estimators(training_vector,clusters_number,segment_size,segments_number)
    return get_predictions(original_vector,estimators,segment_size,segments_number)
'''

class PQ:
    def __init__(self,number_of_clusters:int,number_of_segments:int,data_length:int):
        assert data_length%number_of_segments==0, "Segment Size must be equal"
        self.number_of_clusters = number_of_clusters
        self.number_of_segments = number_of_segments
        self.data_length = data_length
        self.segment_size = self.data_length//self.number_of_segments
        self.isTrained = False
        self.estimators = [KMeans(n_clusters=number_of_clusters,random_state=42) for _ in range(self.number_of_segments)]

    def train(self,training_data:list):
        assert self.isTrained==False, "Estimators are already Trained"
        assert(training_data.shape[1]== self.data_length), "Training Data must have same size of data length"
        training_data = training_data.reshape(self.number_of_segments, training_data.shape[0], self.segment_size) 
        for index,segment in enumerate(training_data):
            self.estimators[index].fit(segment.reshape(-1,self.segment_size))

    def get_compressed_data(self,given_vector:list):
        given_vector = given_vector.reshape(self.number_of_segments, self.segment_size)
        result_vector=[]
        for i,estimator in enumerate(self.estimators):
            segment_predictions = estimator.predict(given_vector[i].reshape(-1, self.segment_size))
            result_vector.append(segment_predictions[0])
        return result_vector

        
if __name__ == '__main__':
    pq_model = PQ(number_of_clusters=64,number_of_segments=4,data_length=12)
    pq_model.train(np.random.random((100,12)))
    for i in range(5):
        vector = np.random.random(12)
        print("Vector :")
        print(vector)
        print("Compressed Vector :")
        print(pq_model.get_compressed_data(vector))
        print("-----")
