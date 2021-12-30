import torch
import scipy.io
import os
import random

from PIL import Image
import numpy as np
from pydoc import locate
from torchvideotransforms import video_transforms, volume_transforms


def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((455,256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def denormalize(label, class_idx, upper = 100.0):
    label_ranges = {
        1: (21.6, 102.6),
        2: (12.3, 16.87),
        3: (8.0, 50.0),
        4: (8.0, 50.0),
        5: (46.2, 104.88),
        6: (49.8, 99.36)}
    label_range = label_ranges[class_idx]

    true_label = (label.float() / float(upper)) * (label_range[1] - label_range[0]) + label_range[0]
    return true_label


def normalize(label, class_idx, upper = 100.0):

    label_ranges = {
        1: (21.6, 102.6),
        2: (12.3, 16.87),
        3: (8.0, 50.0),
        4: (8.0, 50.0),
        5: (46.2, 104.88),
        6: (49.8, 99.36)}
    # print('key: ', class_idx)
    label_range = label_ranges[class_idx]

    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0]) ) * float(upper)
    return norm_label

class SevenPair_all_Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, class_idx_list, data_root, frame_length, subset, score_range=100, num_exemplar=1):
        random.seed(0)
        self.transforms = transform
        classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        self.classes_name = classes_name
        self.sport_classes = [classes_name[class_idx - 1] for class_idx in class_idx_list]
        self.class_idx_list = class_idx_list
        self.score_range = score_range
        self.subset = subset
        self.num_exemplars = num_exemplar
        # file path
        self.data_root = data_root
        self.data_path = [os.path.join(self.data_root, '{}-out'.format(sport_class)) for sport_class in
                          self.sport_classes]
        self.split_path = os.path.join(self.data_root, 'Split_4', 'split_4_train_list.mat')
        self.split = scipy.io.loadmat(self.split_path)['consolidated_train_list']
        self.split = [item.tolist() for item in self.split if item[0] in self.class_idx_list]
        if self.subset == 'test':
            self.split_path_test = os.path.join(self.data_root, 'Split_4', 'split_4_test_list.mat')
            self.split_test = scipy.io.loadmat(self.split_path_test)['consolidated_test_list']
            self.split_test = [item.tolist() for item in self.split_test if item[0] in self.class_idx_list]

        # setting
        self.length = frame_length

        if self.subset == 'test':
            self.dataset = self.split_test.copy()
        else:
            self.dataset = self.split.copy()

    def load_video(self, idx, action_class):
        # self.data_path = [os.path.join(self.data_root, '{}-out'.format(sport_class)) for sport_class in
        #                   self.sport_classes]
        # video_path = os.path.join(self.data_root, '{}-out'.format(self.classes_name[action_class - 1]), '%03d' % idx)
        video_path = os.path.join(self.data_root, 'frames/{}'.format(self.classes_name[action_class - 1]), '%03d' % idx)   # for isee
        # video = [Image.open(os.path.join(video_path, 'img_%05d.jpg' % (i + 1))) for i in range(self.length)]
        video = [Image.open(os.path.join(video_path, 'image_%05d.jpg' % (i + 1))) for i in range(self.length)]    # for isee
        return self.transforms(video)

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        action_class = int(sample_1[0])
        idx = int(sample_1[1])
        data_1 = {}

        # 暂且先将测试集合训练集的数据设为一致的
        if self.subset == 'test':
            # test phase
            data_1['video'] = self.load_video(idx, action_class)
            data_1['real_score'] = sample_1[2]
            data_1['final_score'] = normalize(sample_1[2], action_class, self.score_range)
            data_1['action_class'] = int(sample_1[0])
            # choose data_2
            # choose a list of sample in training_set
            exemplar_list = []
            for i in range(self.num_exemplars):
                data_2 = {}
                train_file_list = self.split.copy()
                tmp_idx = random.randint(0, len(train_file_list) - 1)
                sample_2 = train_file_list[tmp_idx]
                action_class_2 = int(sample_2[0])
                data_2['video'] = self.load_video(int(sample_2[1]),action_class_2)
                data_2['real_score'] = sample_2[2]
                data_2['final_score'] = normalize(sample_2[2], action_class_2, self.score_range)
                data_2['action_class'] = int(sample_2[0])
                exemplar_list.append(data_2)
            return data_1, exemplar_list
        else:
            # train phase
            # load video：get all the frams of the video and do transformation
            data_1['video'] = self.load_video(idx, action_class)
            data_1['real_score'] = sample_1[2]
            data_1['final_score'] = normalize(sample_1[2], action_class, self.score_range)
            data_1['action_class'] = int(sample_1[0])
            # choose data_2
            # choose a list of sample in training_set
            data_2 = {}
            train_file_list = self.split.copy()
            tmp_idx = random.randint(0, len(train_file_list) - 1)
            sample_2 = train_file_list[tmp_idx]
            action_class_2 = int(sample_2[0])
            data_2['video'] = self.load_video(int(sample_2[1]), action_class_2)
            data_2['real_score'] = sample_2[2]
            data_2['final_score'] = normalize(sample_2[2], action_class_2, self.score_range)
            data_2['action_class'] = int(sample_2[0])
            return data_1, [data_2]


    def __len__(self):
        return len(self.dataset)
