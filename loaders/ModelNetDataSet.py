'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNet40DataSet(Dataset):
    class_num = 40

    def __init__(self, root, train=True, points=1024, use_uniform_sample=True, process_data=False, **kwargs):
        """
        :param root: root path to dataset, e.g. data/modelnet40_normal_resampled/
        :param train: train or test, default True
        :param points: default sampled points
        :param use_uniform_sample: True: FPS; False: first points selected
        :param process_data: if preprocess data
        :param kwargs:
        """
        self.root = root
        class_names_file = "modelnet%s_shape_names.txt" % (self.class_num)
        self.class_names = [line.rstrip() for line in open(os.path.join(self.root, class_names_file))]
        self.train = train
        self.points = points
        self.use_uniform_sample = use_uniform_sample
        if self.train:
            self.datafile = "modelnet%s_train.txt" % (self.class_num)
        else:
            self.datafile = "modelnet%s_test.txt" % (self.class_num)
        temp_Train = "train" if self.train else "test"
        temp_uniform = "FPS" if self.use_uniform_sample else "RND"
        self.processed_file = "modelnet%s_%s_%spts_%s.npz" % (self.class_num, temp_Train, self.points, temp_uniform)
        if process_data:
            print(f"===> processing data: {self.processed_file}")
            if os.path.isfile(os.path.join(self.root, self.processed_file)):
                print(f"===> File {self.processed_file} exsits, ignoring processing.")
            else:
                self._process_data()

        # now load the picked numpy arrays
        processed_data = np.load(os.path.join(self.root, self.processed_file))
        self.data = processed_data["data"]
        self.targets = processed_data["targets"]

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        data[:, 0:3] = pc_normalize(data[:, 0:3])
        return data, target

    def __len__(self):
        return len(self.data)

    def _process_data(self):
        points = []
        targets = []
        file_names = [line.rstrip() for line in open(os.path.join(self.root, self.datafile))]
        for i in tqdm(range(len(file_names))):
            file = file_names[i]
            # class_temp = file.split("_")[0]
            class_temp = file[:-5]
            file_data = np.genfromtxt(os.path.join(self.root, class_temp, file + '.txt'),
                                      delimiter=',').astype(np.float32)
            file_target = np.int32(self.class_names.index(class_temp))
            if self.use_uniform_sample:  # which indicates the farthest point sampling
                file_data = farthest_point_sample(file_data, self.points)
            else:
                file_data = file_data[0:self.points, :]
            points.append(file_data)
            targets.append(file_target)
        # return points, targets
        np.savez_compressed(os.path.join(self.root, self.processed_file), data=points, targets=targets)


if __name__ == '__main__':
    modelnet40_train_dataset = ModelNet40DataSet(root="/work/xm0036/data/modelnet40_normal_resampled",
                                   train=True, points=1024, use_uniform_sample=True, process_data=True)
    modelnet40_test_dataset = ModelNet40DataSet(root="/work/xm0036/data/modelnet40_normal_resampled",
                                   train=False, points=1024, use_uniform_sample=True, process_data=True)
    print(modelnet40_train_dataset.__len__())
    print(modelnet40_test_dataset.__len__())
