import pandas as pd
from kmeans import *

PATH = "saved_db.csv"


def read_data():
    # Generate random data (replace this with your actual data)
    # Read vectors from CSV file
    df = pd.read_csv(PATH)

    # Extract the vectors from the DataFrame
    return df.values


# Dimensionality Division
def layer1(data):
    return data


# Clustering/Centroids
def layer2(data):
    run_kmeans(data)


# HNSW
def layer3(data):
    return data


def index():
    vectors = read_data()
    layer1_output = layer1(vectors)
    layer2_output = layer2(layer1_output)
    result = layer3(layer2_output)
    print(result)


if __name__ == '__main__':
    index()
