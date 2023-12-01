import math
from sklearn.cluster import KMeans
import numpy as np





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

if __name__ == '__main__':
    vector = np.random.random(12)
    print(vector)
    training_vector = np.random.random((50,12))
    print(PQ(vector,training_vector,4,8))