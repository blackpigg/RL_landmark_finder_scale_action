#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 23:31:43 2017

@author: wd
"""
import numpy as np
import gym
from gym import utils
from gym import spaces
import h5py
import tensorflow as tf
import random
import scipy.misc


class NDEnv(gym.Env):
    # ======================================================================================================================
    # load image, label and define scale of env space
    def __init__(self, desc=None, img_index=random.randint(0, 9995), length=28, scale_step=0.3):

        self.imdb = h5py.File('/home/wd/Downloads/imdb_100_2.mat')
        self.image = self.imdb.get('imdb/images/data')
        self.label = self.imdb.get('imdb/images/labels')

        self.image = np.asarray(self.image)
        # normalize image
        self.image = (self.image - 127.5) / 127.5
        self.label = np.asarray(self.label)

        # define scale of agent
        self.width = length
        self.height = length

        # define scale of environment
        self.scale_factor = 1.3
        self.scale = 1
        self.nrow = np.shape(self.image[0])[1] * self.scale_factor
        self.ncol = np.shape(self.image[0])[1] * self.scale_factor

        self.scale_list = [int(length * scale) for scale in np.arange(1, int(self.nrow/length), scale_step)]
        self.scale_idx = 0
        self.scale_step = scale_step

        # define number of action: up, down, left, right, trigger
        self.action_space = spaces.Discrete(7)
        self._reset()
        self.get_cover([0, 0, 1, 1], [10, 10, 20, 20])


    # =======================================================================================================================
    # initialize environment and agent
    def _reset(self):

        # index of image at dataset
        self.idx = random.randint(0, 9995)
        self.desc = self.image[self.idx, :, :, :]
        # rotate image
        self.desc = np.moveaxis(self.desc, 1, 2)
        # resize image
        self.desc = scipy.misc.imresize(self.desc[0], (130, 130))

        # get label of image
        self.eye_label = self.label[self.idx, 0:2]
        self.nose_label = self.label[self.idx, 4:6]

        # scaling label
        self.gt_x = int(self.eye_label[0] * self.scale_factor)
        self.gt_y = int(self.eye_label[1] * self.scale_factor)

        gt_box_scale = 0.5
        self.gt_box = [self.gt_x - gt_box_scale * self.width, self.gt_y - gt_box_scale * self.height,
                       self.gt_x + gt_box_scale * self.width, self.gt_y + gt_box_scale * self.height]
        # initialize point of agent
        # self.x, self.y are upper-left point of agent
        x = np.random.randint(low=max(0, self.gt_x - 1.5 * gt_box_scale * self.width - 13.5), \
                                   high=min(self.gt_x + 1.5 * gt_box_scale * self.width - 13.5, self.nrow - self.width))

        y = np.random.randint(low=max(0, self.gt_y - 1.5 * gt_box_scale * self.width - 13.5), \
                                   high=min(self.gt_y + 1.5 * gt_box_scale * self.width - 13.5, self.nrow - self.width))

        # state_pt : center point of agent
        self.cur_x = int(x)#+0.5*self.width
        self.cur_y = int(y)#+0.5*self.height
        self.state = np.reshape(self.desc[self.cur_y:self.cur_y + self.scale_list[0],
                                self.cur_x:self.cur_x + self.scale_list[0]], (1, 784))
        self.scale_idx = 1


        return self.state[0], self.cur_x, self.cur_y, self.gt_x, self.gt_y, self.scale_list[self.scale_idx], self.desc

    # =======================================================================================================================
    # define how state move
    def step(self, action):

        # trigger: 0
        # right: 1
        # down: 2
        # left: 3
        # up: 4
        # expand : 5
        # shrink : 6
        # new_state = self.state[0]
        scale_idx = self.scale_idx
        x = self.cur_x
        y = self.cur_y
        if np.argmax(action) == 0:  # trigger
            pass
        elif np.argmax(action) == 1:  # right
            x = min(x + 1, self.nrow - self.width - 2)

        elif np.argmax(action) == 2:  # down
            y = min(y + 1, self.ncol - self.height - 2)

        elif np.argmax(action) == 3:
            x = max(x - 1, 0)

        elif np.argmax(action) == 4:  # up
            y = max(y - 1, 0)

        elif np.argmax(action) == 5:  # expand
            new_scale_idx = min(len(self.scale_list)-1, scale_idx + 1)
            x, y = self.adjust_pos(x, y, scale_idx, new_scale_idx)
            scale_idx = new_scale_idx

        elif np.argmax(action) == 6:  # shrink
            new_scale_idx = max(0, scale_idx - 1)
            x, y = self.adjust_pos(x, y, scale_idx, new_scale_idx)
            scale_idx = new_scale_idx

        x, y = int(x), int(y)
        new_state = self.getnewstate(x, y, scale_idx)


        new_state_pt_box = [x, y, x+self.scale_list[scale_idx], y, y+self.scale_list[scale_idx]]

        if np.argmax(action) == 0 and self._get_IoU(self.gt_box, new_state_pt_box) > 0.7:
            done = 1
            reward = 1


        elif np.argmax(action) == 0 and self._get_IoU(self.gt_box, new_state_pt_box) < 0.7:
            done = 1
            reward = -1

        else:
            reward = 0
            done = 0
        self.scale_idx = scale_idx

        shape = np.shape(new_state)
        a = shape[0]
        b = shape[1]
        if a != b:
            print(np.shape(new_state))
            print(np.argmax(action))
            print(x)
            print(y)
            print(self.cur_x)
            print(self.cur_y)
            print(self.scale_list[scale_idx])
            print(self.scale_list[scale_idx-1])

        self.cur_x = x
        self.cur_y = y

        new_state = np.reshape(scipy.misc.imresize(new_state, (28, 28)), (1, 784))

        return new_state, reward, done, x, y, self.scale_list[scale_idx]

    # =======================================================================================================================

    # get IoU of two box
    def _get_IoU(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x1 > x2 and y1 > y2:
            interArea = -1 * (x2 - x1 + 1) * (y2 - y1 + 1)
        else:
            interArea = (x2 - x1 + 1) * (y2 - y1 + 1)

        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        if interArea < 0:
            iou = interArea / float(box1Area + box2Area)
        else:
            iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    def get_cover(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        gt_box = (box2[2]-box2[0]+1)**2


        if x1 > x2 and y1 > y2:
            interArea = 0
        else:
            interArea = (x2 - x1 + 1) * (y2 - y1 + 1)

        return interArea/gt_box


    # return binary value : value is 1 if  box2 is subset of box1
    #                      else value is 0
    def _inGT(self, box1, box2):
        x1 = box1[0] - box2[0]
        y1 = box1[1] - box2[1]
        x2 = box1[2] - box2[2]
        y2 = box1[3] - box2[3]

        if x1 < 0 and y1 < 0 and x2 > 0 and y2 > 0:
            result = 1
        else:
            result = 0
        return result

        # return distance of boxes

    def _get_distance(self, box1, box2):
        dis = np.sqrt((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2)
        return dis

    def getnewstate(self, cur_x, cur_y, scale_idx):
        new_state = self.desc[cur_y:cur_y + self.scale_list[scale_idx], cur_x:cur_x + self.scale_list[scale_idx]]
        return new_state

    def adjust_pos(self, cur_x, cur_y, scale_idx, new_scale_idx):

        if new_scale_idx > scale_idx:
            new_x = cur_x - 0.5 * (self.scale_list[new_scale_idx] - self.scale_list[scale_idx]) - 1
            new_y = cur_y - 0.5 * (self.scale_list[new_scale_idx] - self.scale_list[scale_idx]) - 1
            left_delta = new_x
            right_delta = self.ncol - (new_x + self.scale_list[scale_idx + 1])
            up_delta = new_y
            down_delta = self.nrow - (new_y + self.scale_list[scale_idx + 1])
            if new_x < 0:
                new_x = 0
            elif right_delta <= 0:
                new_x += right_delta
            if new_y < 0:
                new_y = 0
            elif down_delta <= 0:
                new_y += down_delta
        else:
            new_x = cur_x + 0.5 * (self.scale_list[scale_idx] - self.scale_list[new_scale_idx])
            new_y = cur_y + 0.5 * (self.scale_list[scale_idx] - self.scale_list[new_scale_idx])

        return new_x, new_y

temp = NDEnv(gym.Env)
