import math
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import bisect
import struct
import time
from PQ import *
from worst_case_implementation import *
from dataclasses import dataclass
from scipy.cluster.vq import kmeans2

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

    def __init__(self, cluster_bits: int, number_of_segments: int, data_length: int,data_path: str,initially_load:bool):
        assert data_length % number_of_segments == 0, "Segment Size must be equal"
        #Number of Clusters per each segment
        self.number_of_clusters = 2**cluster_bits
        #Number of Segments in each database vector
        self.number_of_segments = number_of_segments
        #Length of the given data
        self.data_length = data_length
        #single Segment Size
        self.segment_size = self.data_length//self.number_of_segments
        #Defines if the model is trained or not
        self.isTrained = False
        #Array of Estimators we have for each segment
        #As Each segment has its own dependent estimator
        self.estimators = [MiniBatchKMeans(
            n_clusters=self.number_of_clusters,n_init=10) for _ in range(self.number_of_segments)]
        # Now We Keep The Centroids of each estimator
        self.centroids = []
        self.table = np.empty(
            (self.number_of_segments, self.number_of_clusters))
        self.data_path = data_path
        self.index_file_path = "index.bin"
        self.centroids_file_path = "centroids.bin"
        self.quantized_vectors_path="quantized.bin"
        self.vectors = []
        self.dtype = np.uint8
        self.dtype_original = np.float32

        if initially_load:
            self.centroids = self.load_centroids()
    
    '''
    Actually this is the first step as we are getting the records and store them
    '''
    def insert_records(self,rows: List[Dict[int, Annotated[List[int],14]]],records_np):
        self.train_before_writing(records_np)
        self.save_vectors(rows)
        self.save_quantized_vectors(rows)
        #self.save_centroids()

    def train_before_writing(self,rows):
        max_length = 100000
        if max_length > len(rows):
            max_length = len(rows)
        self.train(rows[:max_length])
        self.trained_data_size = len(rows)

    def save_quantized_vectors(self,records):
        embed_values = np.array([record["embed"] for record in records])
        id_values = np.array([record["id"] for record in records])
        self.get_compressed_data(embed_values)
        with open(self.data_path, 'wb') as file:
            for index,quantized_record in enumerate(self.quantized_vectors):
                id_size = 'i'
                vec_size = 'i' * len(quantized_record)
                binary_data = struct.pack(
                    id_size + vec_size, id_values[index], *quantized_record)
                file.write(binary_data)


    def save_vectors(self,records):
        with open(self.data_path, 'wb') as file:
            for record in records:
                id_size = 'i'
                vec_size = 'f' * len(record["embed"])
                binary_data = struct.pack(
                    id_size + vec_size, record["id"], *record["embed"])
                file.write(binary_data)

    def save_centroids(self):
        with open(self.centroids_file_path, 'wb') as file:
            for centroid in self.centroids:
                vec_size = 'f' * len(centroid)*len([centroid[0]])
                binary_data = struct.pack(vec_size, *centroid)
                file.write(binary_data)



    def load_centroids(self):
        with open(self.centroids_file_path, 'rb') as file:
            data = np.fromfile(file, dtype=np.int32, count=self.number_of_segments * (self.number_of_clusters)*self.segment_size)
            data = data.reshape((self.number_of_segments, (self.number_of_clusters)*self.segment_size))
            print(data)

        pass

    def load_vectors(self):
        pass

    def load_quantized_vectors(self):
        pass
    

    '''
    Here we are reading our data and training our kmeans models on the read data
    So This is The First Step to do
    '''
    def read_data(self):
        chunk_size = struct.calcsize('i') + (struct.calcsize('i') * self.number_of_segments)
        with open(self.data_path, 'rb') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                _, *vector = struct.unpack('I' + 'i' * self.number_of_segments, chunk)
                self.vectors.append(vector)
        self.vectors = np.array(self.vectors)
        self.train(self.vectors)

    
    '''
    Now We should train our estimators before starting any prediction
    '''
    def calculate_distance(self,vec1,vec2):
       return np.linalg.norm(vec1 - vec2)
    

    def calculate_manhattan_distance(self,vec1,vec2):
        return np.sum(np.abs(vec1-vec2))

    def train(self, training_data: list):
        assert self.isTrained == False, "Estimators are already Trained"
        assert (training_data.shape[1] == self.data_length), "Training Data must have same size of data length"
        for index in range(self.number_of_segments):
            print("Training Kmeans model number")
            print(index+1)
            self.estimators[index].fit(training_data[:,index*self.segment_size:(index+1)*self.segment_size])
            self.centroids.append(self.estimators[index].cluster_centers_)
        self.isTrained = True

    '''
    Generates a compressed Vector using The trained PQ
    ''' 

    def get_compressed_data(self, given_vector):
        assert self.isTrained == True, "You Should Train The Models First"
        result_vector = np.zeros((len(given_vector),self.number_of_segments))
        for index in range(self.number_of_segments):
            current_estimator = self.estimators[index]
            current_segment = given_vector[:,index*self.segment_size:(index+1)*self.segment_size]
            result_vector[:,index] = current_estimator.predict(current_segment)
        self.quantized_vectors = result_vector.astype(int)

    '''
    Generate the table between the upcoming vector to estimate its similarity with the existing vectors
    '''

    def generate_query_table(self, query_vector):
        assert self.isTrained == True, "You Should Train The Models First"
        for index in range(self.number_of_segments):
            current_segment = query_vector[0,index*self.segment_size:(index+1)*self.segment_size]
            current_centroids = self.centroids[index]
            for j,centroid in enumerate(current_centroids):
                self.table[index][j]= self.calculate_distance(current_segment,centroid)
            #distances = np.apply_along_axis(self.calculate_distance, 1, current_centroids, current_segment)
            #self.table[index, :] = distances

        # looping over each segment of the given vector
        # taking each segment -> and looping over all the centroids in that segment

    '''
    Get distance between a database vector and the query which we calculated its table before
    '''

    def get_distance(self,query_vector,db_vectors):
        self.generate_query_table(query_vector)
        distance_array = np.zeros((len(self.quantized_vectors))).astype(float)
        for index in range(self.number_of_segments):
            distance_array += self.table[index,db_vectors[:,index]]
        return distance_array
    

    def get_symm_distance(self,query_vector,database_vector):
        self.centroids = np.array(self.centroids)
        query_vector = np.array(query_vector)
        database_vector = np.array(database_vector)
        distances = np.sum(np.abs(self.centroids[np.arange(self.number_of_segments), query_vector] -
                               self.centroids[np.arange(self.number_of_segments), database_vector - 1]), axis=0)
        total_distance = np.sum(distances)
        return total_distance
    

    def retrieve(self,query,top_k):
        print("hn retreive ahu")

        nearest_vectors=[]

        distances = np.empty(self.trained_data_size, dtype=np.float32)
        ids = np.empty(self.trained_data_size, dtype=np.int32)
        distances = None

        with open('saved_db.bin', 'rb') as file:
            data = np.fromfile(file, dtype=np.int32, count=self.trained_data_size * (self.number_of_segments + 1))
            data = data.reshape((self.trained_data_size, self.number_of_segments + 1))

            ids[:] = data[:, 0]
            vectors = data[:, 1:]
            distances = self.get_distance(query,vectors)

        indices = np.argsort(distances)[:top_k]

        return indices
    
