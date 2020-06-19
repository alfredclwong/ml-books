import os
import requests
from tqdm import tqdm
from io import BytesIO
import gzip
import numpy as np
from PIL import Image


def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='big')


def get_gz_obj(url):
    print(f'{url}')
    gz_obj = BytesIO()
    with requests.get(url, stream=True) as response:
        file_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            gz_obj.write(data)
    progress_bar.close()
    gz_obj.seek(0)
    return gz_obj


def get_mnist():
    image_urls = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    }
    label_urls = {
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    }

    print('getting images')
    for fname, url in image_urls.items():
        gz_obj = get_gz_obj(url)
        with gzip.open(gz_obj) as gz_file:
            info = [bytes_to_int(gz_file.read(4)) for i in range(4)]
            magic_number, n_images, n_rows, n_cols = info
            images_bytes = gz_file.read()
            images = np.frombuffer(images_bytes, dtype=np.uint8)
            images = np.reshape(images, (n_images, n_cols, n_rows))
            np.save(os.path.join('data/mnist/', fname), images)

    print('getting labels')
    for fname, url in label_urls.items():
        gz_obj = get_gz_obj(url)
        with gzip.open(gz_obj) as gz_file:
            info = [bytes_to_int(gz_file.read(4)) for i in range(2)]
            magic_number, n_items = info
            labels_bytes = gz_file.read()
            labels = np.frombuffer(labels_bytes, dtype=np.uint8)
            np.save(os.path.join('data/mnist/', fname), labels)


if __name__ == '__main__':
    get_mnist()
