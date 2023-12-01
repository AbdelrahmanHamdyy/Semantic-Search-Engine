import math
'''
This function is used to split the given vector to equally sized m segments
'''
def splittingVector(original_vector,num_of_segments):
    segments = []
    segment_size = math.ceil(len(original_vector)/num_of_segments)
    for index in range(num_of_segments):
        last_segment_index = (index+1)*segment_size
        if last_segment_index > len(original_vector):
            last_segment_index = len(original_vector)
        current_segment = original_vector[index*segment_size:last_segment_index]
        segments.append(current_segment)
    return segments


if __name__ == '__main__':
    vector = [1,2,3,6,5,4,9,8,98,95,6,2,31,5,1,87,1,5,1,3,13,1,31]
    print(splittingVector(vector,6))
