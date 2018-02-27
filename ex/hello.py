from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Step 1 prepare the data


def vectorize_sequences(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1.
    return result


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

