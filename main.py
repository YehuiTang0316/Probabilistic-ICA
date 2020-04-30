import numpy as np
import sklearn
import os
import struct
import matplotlib.pyplot as plt


def load_data(path, dataset, mode):
    """
    :param path: root path of the dataset
    :param mode: mode = ('train', 't10k')
    :return: an array of flatten image
    """
    assert mode == 'train' or 'test'

    data_path = os.path.join(path, dataset, mode)

    if dataset == 'mnist':
        labels_path = os.path.join(data_path, '%s-labels-idx1-ubyte' % mode)
        images_path = os.path.join(data_path, '%s-images-idx3-ubyte' % mode)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return labels, images


def show_images(labels, images):
    # show first 30 images
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(30):
        imgs = np.reshape(images[i], [28, 28])
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(imgs, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(labels[i]))
    plt.show()


def compute_log_likelihood(data, mixing_matrix, unmixing_matrix):
    return np.sum(np.log(0.5 / np.cosh(np.dot(unmixing_matrix, data.T)) ** 2.0), axis=0) \
           + np.log(np.abs(np.linalg.det(mixing_matrix)))


if __name__ == '__main__':
    path = '/Users/tangyehui/UG_Project/dataset'
    train_labels, train_images = load_data(path, 'mnist', 'train')
    test_labels, test_images = load_data(path, 'mnist', 't10k')

    # mask = [_ == 5 for _ in train_labels]
    # train_labels = train_labels[mask]
    # train_images = train_images[mask]
    # show_images(train_labels, train_images)

    # whitening part
    from whitening import Whitening
    w1 = Whitening(train_images[:1000])

    train_images_zca = w1.ZCA_cor_whitening()
    # print(np.linalg.det(np.matmul(w1.G, w1.G.T)))
    # show_images(train_labels, train_images_zca)
    # train_images_zca = np.transpose(train_images_zca)

    # conduct ICA
    from sklearn.decomposition import FastICA
    ica = FastICA(max_iter=200, n_components=784, whiten=False)
    s = ica.fit(train_images_zca).transform(train_images_zca)
    x = ica.fit_transform(train_images_zca)
    recovered = ica.inverse_transform(x)

    # print(x.shape)

    # show_images(train_labels, recovered)
    print(ica.components_.shape)
    print(ica.mixing_.shape)
    print(s.shape)
    print(recovered.shape)

    # show_images(train_labels, s)
    print(np.mean(compute_log_likelihood(test_images, ica.mixing_, ica.components_)))




