"""
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
"""
import os

import cv2
import numpy as np


class Dset(object):
    def __init__(self, names, inputs, labels, randomize):
        self.names = list(names)
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.names)

        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.pointer = None

        self.__init_pointer()

    def __init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx]
            self.labels = self.labels[idx]
            self.names = [self.names[i] for i in idx]

    def get_next_batch(self, batch_samples):
        # if batch_size is negative -> return all
        if isinstance(batch_samples, int):
            batch_size = batch_samples
            if batch_size < 0:
                return self.inputs, self.labels

            if self.pointer + batch_size >= self.num_pairs:
                self.__init_pointer()
            end = self.pointer + batch_size
            inputs = self.inputs[self.pointer:end]
            labels = self.labels[self.pointer:end]
            self.pointer = end
            return inputs, labels

        idx = []
        for name in batch_samples:
            idx.append(-1 if name not in self.names else self.names.index(name))
        inputs = self.inputs[idx]
        labels = self.labels[idx]
        return inputs, labels


class Mujoco_Dset(object):
    # def __init__(self, expert_path, picture_size, randomize=True):
    #     width, height, channel = picture_size
    #
    #     frames, nums, actions = [], [], []
    #     for dir_or_file in os.listdir(expert_path):
    #         picture_path = os.path.join(expert_path, dir_or_file)
    #
    #         if not dir_or_file.endswith('.jpg'):
    #             print(dir_or_file)
    #             continue
    #
    #         frame = cv2.imread(picture_path, cv2.IMREAD_COLOR)
    #         frame = cv2.resize(frame, (width, height))
    #         frames.append(frame)
    #
    #         [num, action, clock] = dir_or_file.split('.')[1].split('_')
    #         nums.append(num)
    #         actions.append([int(action), int(clock)])
    #
    #     frames_array = np.array(frames)
    #     actions_array = np.array(actions)
    #
    #     print(frames_array.shape)
    #     print(nums)
    #     print(actions_array)
    #     self.dset = Dset(nums, frames_array, actions_array, randomize)

    def __init__(self, expert_path, picture_size, randomize=True):
        data = np.load(expert_path)
        nums = list(data['num'])
        print(nums)
        frames = data['obs']
        actions = data['acs']
        print(actions)
        self.dset = Dset(nums, frames, actions, randomize)

    def get_next_batch(self, batch_samples):
        return self.dset.get_next_batch(batch_samples=batch_samples)

    def get_one_batch(self, num):
        obses, actions = self.dset.get_next_batch(batch_samples=[num])
        return obses[0], actions[0]


# class Mujoco_Dset(object):
#     def __init__(self, expert_path, picture_size, show_image=False, frame_rate=20/74, randomize=True):
#         frames_list, nums, actions = [], [], []
#         for dir_or_file in os.listdir(expert_path):
#             video_path = os.path.join(expert_path, dir_or_file)
#             if not dir_or_file.endswith('.avi'):
#                 print(dir_or_file)
#                 continue
#
#             # print('Open: ', video_path)
#             frames = []
#             video = cv2.VideoCapture(video_path)
#             while video.isOpened():
#                 success, frame = video.read()
#
#                 if success:
#                     frame = self.preprocess_image(frame, picture_size, show=show_image)
#                     frames.append(frame)
#
#                     frame_index = video.get(cv2.CAP_PROP_POS_FRAMES)
#                     video_frame_rate = video.get(cv2.CAP_PROP_FPS)
#                     video.set(cv2.CAP_PROP_POS_FRAMES, frame_index + video_frame_rate // frame_rate)
#                     last_frame_index = video.get(cv2.CAP_PROP_FRAME_COUNT)
#                     # print(frame_index, video_frame_rate, last_frame_index)
#
#                     if frame_index >= last_frame_index:  # Video is over
#                         break
#                 else:
#                     break
#
#             frames_list.append(frames[-1])
#             [num, action] = dir_or_file.split('.')[1].split('_')
#             nums.append(num)
#             actions.append(int(action))
#
#         cv2.waitKey(1) & 0xFF
#         cv2.destroyAllWindows()
#
#         frames_array = np.array(frames_list)
#         actions_array = np.array(actions)
#
#         print(frames_array.shape)
#         print(nums)
#         self.dset = Dset(nums, frames_array, actions_array, randomize)
#
#     def preprocess_image(self, image, size, show=False):
#         """ Changes size, makes image monochromatic """
#         width, height, channel = size
#         image = cv2.resize(image, (width, height))
#         if channel == 2:
#             color_style = cv2.COLOR_BGR2GRAY
#         else:
#             color_style = cv2.IMREAD_COLOR
#         image = cv2.cvtColor(image, color_style)
#         image = np.array(image, dtype=np.uint8)
#
#         if show:
#             cv2.imshow('figure', image)
#             cv2.waitKey(1)
#         return image
#
#     def get_next_batch(self, batch_samples):
#         return self.dset.get_next_batch(batch_samples=batch_samples)
#
#     def get_one_batch(self, num):
#         obses, actions = self.dset.get_next_batch(batch_samples=[num])
#         return obses[0], actions[0]
