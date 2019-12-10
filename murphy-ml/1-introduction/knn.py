import numpy as np
from tqdm import tqdm
from scipy import stats


def knn(k, train_data, train_labels, test_data):
    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    pred_labels = np.zeros(n_test, dtype=int)
    for i in tqdm(range(n_test)):
        distances = np.sum(np.square(train_data - test_data[i]), axis=-1)
        nearest_neighbours = np.argsort(distances)[:k]
        nearest_neighbour_classes = train_labels[nearest_neighbours]
        pred_labels[i] = stats.mode(nearest_neighbour_classes)[0]
    return pred_labels


if __name__ == '__main__':
    data_dir = '../../data/mnist/'
    train_images = np.load(f'{data_dir}train_images.npy')
    train_labels = np.load(f'{data_dir}train_labels.npy')
    test_images = np.load(f'{data_dir}test_images.npy')#[:1000]
    test_labels = np.load(f'{data_dir}test_labels.npy')#[:1000]

    n_train = train_images.shape[0]
    n_test = test_images.shape[0]
    train_data = train_images.reshape(n_train, -1).astype(np.float32)
    test_data = test_images.reshape(n_test, -1).astype(np.float32)

    pred_labels = knn(1, train_data, train_labels, test_data)

    n_correct = np.sum(test_labels == pred_labels)
    print(f'{n_correct/n_test*100:.2f}%')
