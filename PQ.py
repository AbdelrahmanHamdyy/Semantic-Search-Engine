import math
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

'''
This Class is used to implement PQ using its all functionalities
'''


class PQ:
    '''
    Firstly What we have to know about the PQ which we will use is 
    -> number of clusters will be used in our k means which will limit the value will be produced for each segment
    -> number of segments that our data will be splitted to
    -> data length which specifies the original size of the vectors we will have
    Then we will get the size of each segment
    And Initializing The K-means estimators that we will use in predicting
    '''

    def __init__(self, cluster_bits: int, number_of_segments: int, data_length: int):
        assert data_length % number_of_segments == 0, "Segment Size must be equal"
        self.number_of_clusters = pow(2, cluster_bits)
        self.number_of_segments = number_of_segments
        self.data_length = data_length
        self.segment_size = self.data_length//self.number_of_segments
        self.isTrained = False
        self.estimators = [MiniBatchKMeans(
            n_clusters=self.number_of_clusters, random_state=42, n_init='auto') for _ in range(self.number_of_segments)]
        # Now We Keep The Centroids of each cluster
        self.centroids = []
        self.table = np.zeros(
            (self.number_of_clusters, self.number_of_segments))
    '''
    Now We should train our estimators before starting any prediction
    '''

    def train(self, training_data: list):
        assert self.isTrained == False, "Estimators are already Trained"
        assert (
            training_data.shape[1] == self.data_length), "Training Data must have same size of data length"
        training_data = training_data.reshape(
            self.number_of_segments, training_data.shape[0], self.segment_size)
        for index, segment in enumerate(training_data):
            self.estimators[index].fit(segment.reshape(-1, self.segment_size))
            self.centroids.append(self.estimators[index].cluster_centers_)
        self.isTrained = True

    '''
    Generates a compressed Vector using The trained PQ
    '''

    def get_compressed_data(self, given_vector: list):
        assert self.isTrained == True, "You Should Train The Models First"
        given_vector = given_vector.reshape(
            self.number_of_segments, self.segment_size)
        result_vector = []
        for i, estimator in enumerate(self.estimators):
            segment_predictions = estimator.predict(
                given_vector[i].reshape(-1, self.segment_size))
            result_vector.append(segment_predictions[0])
        return result_vector

    '''
    Generate the table between the upcoming vector to estimate its similarity with the existing vectors
    '''

    def generate_query_table(self, query_vector: list):
        assert self.isTrained == True, "You Should Train The Models First"
        vector_segments = query_vector.reshape(-1, self.segment_size)
        # looping over each segment of the given vector
        # taking each segment -> and looping over all the centroids in that segment
        for i in range(self.number_of_segments):
            for j in range(self.number_of_clusters):
                self.table[j][i] = np.linalg.norm(
                    vector_segments[i] - self.centroids[i][j])

    '''
    Get distance between a database vector and the query which we calculated its table before
    '''

    def get_distance(self, database_vector: list):
        sum = 0
        for i in range(self.number_of_segments):
            sum += pow(self.table[database_vector[i]-1][i], 2)
        return pow(sum, 0.5)


if __name__ == '__main__':
    pq_model = PQ(cluster_bits=3, number_of_segments=5, data_length=20)
    training_vectors = np.random.random((100, 20))
    pq_model.train(training_vectors)
    vector = np.random.random(20)
    print("Vector :")
    print(vector)
    print("Compressed Vector :")
    print(pq_model.get_compressed_data(vector))
    print("-----")
    pq_model.generate_query_table(vector)
    for training_vector in (training_vectors):
        print("Original Vector: ")
        print(vector)
        print("training Vector: ")
        print(training_vector)
        print("Distance")
        print(pq_model.get_distance(
            database_vector=pq_model.get_compressed_data(training_vector)))
        print("---------------")
