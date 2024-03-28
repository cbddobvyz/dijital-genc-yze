import random
import numpy

from typing import List
from scipy.stats import skewnorm
from sklearn.datasets import make_blobs


def generate_skewed_data(location: float = 1.0,
                         skewness: float = 0.5,
                         size: int = 10000,
                         scale: float = 1,
                         floor: float = 0.0):
    gen_data = skewnorm.rvs(a=skewness,
                            loc=location,
                            size=size,
                            scale=1.0)
    gen_data = gen_data - min(gen_data)
    gen_data = gen_data / max(gen_data)
    gen_data = gen_data * scale
    gen_data = gen_data + floor
    return gen_data


def add_noise(data_point: float,
              gaussian: [float, float]):
    mean_shift = (random.random() * gaussian[0]) * random.choice([-1, 1])
    return numpy.random.normal(loc=data_point + mean_shift,
                               scale=gaussian[1])


def generate_skewed_data_left(size: int = 10000):
    return numpy.power(generate_skewed_data(100.0,
                                            skewness=-5,
                                            size=size),
                       0.75)


def generate_skewed_data_right(size: int = 10000):
    return numpy.power(generate_skewed_data(100.0,
                                            skewness=5,
                                            size=size),
                       1.0)


def generate_extreme_skewed_data_right(size: int = 10000,
                                       scale: int = 1):
    return numpy.power(generate_skewed_data(100.0,
                                            skewness=5,
                                            size=size,
                                            scale=scale),
                       3.0)


def generate_uniform_dataset(n_data_points=100,
                             feature_range: int = 1.0,
                             random_state: int = None):
    if random_state is None:
        random_state = int(random.random() * 100)
    return numpy.random.RandomState(random_state).uniform(0,
                                                          feature_range,
                                                          size=(n_data_points, 2))


def generate_blobbed_dataset(n_data_points=100,
                             feature_range: int = 1.0,
                             n_blobs: int = 3,
                             blob_std: float = None,
                             blob_centers: List[List[float]] = None,
                             size_blobs: List[int] = None,
                             random_state: int = None):
    if random_state is None:
        random_state = int(random.random() * 100)
    if size_blobs is None:
        size_blobs = [int(random.random() * 10) for x in range(n_blobs)]
    if blob_std is None:
        blob_std = random.random() * 0.2 * feature_range
    if blob_centers is None:
        blob_centers = [[random.random() * feature_range,
                         random.random() * feature_range] for x in range(n_blobs)]
    blob_samples = []
    for blob_counter in range(len(size_blobs)):
        blob_samples.append(n_data_points * size_blobs[blob_counter] // sum(size_blobs))
    return make_blobs(n_samples=blob_samples,
                      cluster_std=blob_std,
                      centers=blob_centers,
                      random_state=random_state)[0]
