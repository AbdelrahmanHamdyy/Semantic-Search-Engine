import math
from sklearn.cluster import KMeans
'''
This function is used to split the given vector to equally sized m segments
'''
def splitting_vector(original_vector,num_of_segments):
    segments = []
    segment_size = math.ceil(len(original_vector)/num_of_segments)
    for index in range(num_of_segments):
        last_segment_index = (index+1)*segment_size
        if last_segment_index > len(original_vector):
            last_segment_index = len(original_vector)
        current_segment = original_vector[index*segment_size:last_segment_index]
        segments.append(current_segment)
    return segments

def get_k_means_estimators(num_of_estimators,num_of_clusters):
    estimators = [KMeans(n_clusters=num_of_clusters,) for _ in range(num_of_estimators)]
    return estimators

'''
Training data should be of size (n,d)
as n is number of training data we have
and d in the size of each of them
so we will split each training data to m segments
and each segment in each row will be used in the training of the corresponding estimator
'''
def train_k_means_estimators(estimators,training_data):
    segment_size = math.ceil(training_data.shape[1]/len(estimators))
    trained_estimators =[]
    for index in range(len(estimators)):
        current_estimator = estimators[index]
        #Select The needed data that will be used in training
        #As we will take all the rows
        #but specific columns
        last_segment_index = (index+1)*segment_size
        if last_segment_index > training_data.shape[1]:
            last_segment_index = training_data.shape[1]
        current_training_data = training_data[:,index*segment_size:last_segment_index]
        current_estimator.fit(current_training_data)
        trained_estimators.append(current_estimator)




if __name__ == '__main__':
    vector = [1,2,3,6,5,4,9,8,98,95,6,2,31,5,1,87,1,5,1,3,13,1,31]
    print(splitting_vector(vector,6))
