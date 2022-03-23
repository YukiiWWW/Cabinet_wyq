import cv2
from hungarian import *
import random
from utils import *
import numpy as np
import math


class Trajectory:
    def __init__(self, id, box, frame, maxPointNum, lineProportion):
        self.id = id
        self.categories = {}
        xywh, category = box
        category = int(category)
        self.categories[category] = self.categories.get(category, 0) + 1
        self.lFrame = frame
        self.cFrame = frame
        self.maxPointNum = maxPointNum
        self.lineProportion = lineProportion
        self.W = self.cFrame.shape[1]
        self.H = self.cFrame.shape[0]

        self.noMatchNum = 0
        self.jStart = False
        self.point_start = xywh
        self.points = [xywh]
        self.points_pm = []
        self.areas = [xywh[2] * xywh[3]]
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.tracker = cv2.TrackerKCF_create()
        initial_p = (self.point_start[0] * self.W, self.point_start[1] * self.H,
                     self.point_start[2] * self.W, self.point_start[3] * self.H)
        self.tracker.init(self.cFrame, initial_p)

        self.category = None
        self.voteCategory()
        # for action judgement
        self.disappear_frames = 0
        self.bind_id = None
        self.valid_length = 1
        self.motionInterval = 5
        self.motionVector = None
        self.det_cnt = 0

    def update(self, frame, xywh_c=None):
        self.lFrame = self.cFrame
        self.cFrame = frame
        if xywh_c is not None:
            xywh, category = xywh_c
            self.noMatchNum = 0
            self.valid_length = len(self.points)  # update valid length if detect the obj
            self.det_cnt += 1
            self.categories[category] = self.categories.get(category, 0) + 1  # maximal category number plus one
        else:
            xywh_l, category = self.getLastBox()
            xywh_l = (xywh_l[0] * self.W, xywh_l[1] * self.H,
                      xywh_l[2] * self.W, xywh_l[3] * self.H)
            self.tracker.init(self.lFrame, xywh_l)
            xywh = self.tracker.update(self.cFrame)[1]
            xywh = [xywh[0] / self.W, xywh[1] / self.H,
                    xywh[2] / self.W, xywh[3] / self.H]
            self.noMatchNum += 1
        self.points.append(xywh)
        self.areas.append(xywh[2] * xywh[3])
        if len(self.points) > self.maxPointNum:
            self.points.pop(0)
            self.areas.pop(0)
        self.length = len(self.points)  # update valid length if detect the obj
        # motion
        # if self.length >= self.motionInterval:
        t_l = -min([self.length, self.motionInterval])
        t_r = -1
        l_point = self.points[t_l]
        c_point = self.points[t_r]
        l_point = [l_point[0] + l_point[2] / 2, l_point[1] + l_point[3] / 2, l_point[2], l_point[3]]
        c_point = [c_point[0] + c_point[2] / 2, c_point[1] + c_point[3] / 2, c_point[2], c_point[3]]
        motionVector = (c_point[0] - l_point[0], c_point[1] - l_point[1])
        length = math.sqrt(motionVector[0] ** 2 + motionVector[1] ** 2)
        if length != 0:
            self.motionVector = (motionVector[0], motionVector[1],
                                 motionVector[0] / length, motionVector[1] / length)

        if len(self.points) <= 3:
            self.voteCategory()

    def getLastBox(self):
        return [self.points[-1], self.category]

    def voteCategory(self):
        max_index = np.argmax(list(self.categories.values()))
        self.category = list(self.categories.keys())[max_index]
        return self.category

    @staticmethod
    def calculateXY(self, p):
        '''
        :param self:
        :param p: [x1, y1, w, h]
        :return: [x, y]
        '''
        return p[0] + p[2] / 2, p[1] + p[3] / 2

    def calculateEnterOrLeave(self):
        # y = ax + b
        p1, p2 = self.lineProportion
        a = (p2[1]-p1[1])/(p2[0]-p1[0])
        b = p1[1] - a * p1[0]

        for p in self.points:
            p = self.calculateXY(p)
            result = a * p[0] + b - p[1]
            if result < 0:
                self.points_pm.append(True)
            else:
                self.points_pm.append(False)


class Matcher:
    '''
    Attention: The category is not considered when matching detection and tracking
    '''
    def __init__(self, lineProportion, maxPointNum=10, maxNoMatchNum=3, class_names=None):
        # initialize parameters
        self.lineProportion = lineProportion
        self.maxPointNum = maxPointNum
        self.maxNoMatchNum = maxNoMatchNum
        self.class_names = class_names
        self.name2class = {cn:i for i, cn in enumerate(self.class_names)}
        self.classCount = {class_name: 0 for class_name in self.class_names}
        self.is_first = True  # first frame initialization flag
        self.lastFrame = None
        self.currentFrame = None

        self.id = 0
        self.trajectories = []
        self.hungarian = Hungarian(is_profit_matrix=True)
        # simple version
        self._appear = []
        self._disappear = []
        # result
        self.takeIn = []
        self.takeOut = []
        self.result = []
        self.thr_l = 0.2
        self.thr_r = 0.8
        self.thr_u = 0.15
        self.thr_m = 0.04
        self.thr_det = 3

    def update(self, img, boxes):
        self._appear = []
        self._disappear = []
        boxes = [[[b[0], b[1], b[2], b[3]], b[-1]] for b in boxes]
        # Using the first frame to init
        if self.is_first:
            self.is_first = False
            self.lastFrame = img
            self.currentFrame = img
            for box in boxes:
                self.id += 1
                self.trajectories.append(
                    Trajectory(self.id, box, self.currentFrame, self.maxPointNum, self.lineProportion))
                self._appear.append(box)
        # frame > 1
        else:
            self.lastFrame = self.currentFrame
            self.currentFrame = img

            last_boxes = []
            for trajectory in self.trajectories:
                last_boxes.append(trajectory.getLastBox())
            l_length = len(last_boxes)  # [[x, y, w, h], category]
            c_length = len(boxes)  # [[x, y, w, h], category]

            if l_length == 0:  # don't have trajectories
                for box in boxes:
                    self.id += 1
                    self.trajectories.append(
                        Trajectory(self.id, box, self.currentFrame, self.maxPointNum, self.lineProportion))
                    self._appear.append(box)
            else:
                if c_length == 0:  # current frame has no objs detected
                    for trajectory in self.trajectories:
                        trajectory.update(self.currentFrame)
                else:
                    last_box_xy = self.translate_xywh(last_boxes)
                    cur_box_xy = self.translate_xywh(boxes)
                    # construct profit matrix
                    profit_matrix = []
                    for l_box in last_box_xy:
                        l_dis = []
                        for c_box in cur_box_xy:
                            l_dis.append(self.calculateDistance((l_box[0] + l_box[2] / 2, l_box[1] + l_box[3] / 2),
                                                                (c_box[0] + c_box[2] / 2, c_box[1] + c_box[3] / 2)))
                        profit_matrix.append(l_dis)

                    self.hungarian.calculate(profit_matrix)
                    h_result = self.hungarian.get_results()
                    l_matched_pair = {m[0]: m[1] for m in h_result if profit_matrix[m[0]][m[1]] < 0.02}
                    c_matched_pair = {m[1]: m[0] for m in h_result if profit_matrix[m[0]][m[1]] < 0.02}
                    # matched
                    for num_i in range(l_length):
                        if num_i in l_matched_pair.keys():
                            box = boxes[l_matched_pair[num_i]]
                            self.trajectories[num_i].update(self.currentFrame, box)
                        else:# unmatched
                            self.trajectories[num_i].update(self.currentFrame)
                    # unmatched
                    for num_i in range(c_length):
                        if num_i not in c_matched_pair.keys():
                            self.id += 1
                            box = boxes[num_i]
                            self.trajectories.append(
                                Trajectory(self.id, box, self.currentFrame, self.maxPointNum, self.lineProportion))
                            self._appear.append(box)
                for num_i, trajectory in enumerate(self.trajectories):
                    if trajectory.noMatchNum > self.maxNoMatchNum:
                        self._disappear.append(self.trajectories[num_i])
                        self.trajectories.pop(num_i)
            # print('trajectory length: {}'.format(len(self.trajectories)))
        self.update_result()

    def translate_xywh(self, boxes, xywh=True):
        '''
        :param boxes: [[[x1, y1, w, h], category], ...]
        :param xywh: True: x1y1wh --> xywh | False: xywh --> x1y1wh
        :return: result
        '''
        boxesReturn = []
        if xywh:
            for box in boxes:
                box = box[0]
                boxTemp = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]]
                boxesReturn.append(boxTemp)
        else:
            for box in boxes:
                box = box[0]
                boxTemp = [box[0] - box[2] / 2, box[1] - box[3] / 2, box[2], box[3]]
                boxesReturn.append(boxTemp)
        return boxesReturn

    def calculateDistance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def update_result(self):
        for num_i, trajectory in enumerate(self.trajectories):
            if not trajectory.jStart and trajectory.motionVector is not None:
                f_inCabinet = self.thr_l < trajectory.point_start[0] + trajectory.point_start[2] / 2 < self.thr_r and \
                              trajectory.point_start[1] + trajectory.point_start[3] / 2 < self.thr_u
                f_motion = trajectory.motionVector[0] > self.thr_m or trajectory.motionVector[1] > self.thr_m
                f_det_cnt = trajectory.det_cnt > self.thr_det
                if f_inCabinet and f_motion and f_det_cnt:
                    trajectory.jStart = True
                    self.takeIn.append(trajectory.category)
        for num_i, trajectory in enumerate(self._disappear):
            last_num = -trajectory.noMatchNum - 1
            f_inCabinet = self.thr_l < trajectory.points[last_num][0] + trajectory.points[last_num][2] / 2 < self.thr_r and \
                          trajectory.points[last_num][1] + trajectory.points[last_num][3] / 2 < self.thr_u
            f_motion = trajectory.motionVector[0] > self.thr_m or trajectory.motionVector[1] > self.thr_m
            f_det_cnt = trajectory.det_cnt > self.thr_det
            if f_inCabinet and f_motion and f_det_cnt:
                self.takeOut.append(trajectory.category)
            self._disappear.pop(num_i)
        self.takeInCate = [self.class_names[int(i)] for i in self.takeIn]
        self.takeOutCate = [self.class_names[int(i)] for i in self.takeOut]
        classCount = {class_name: 0 for class_name in self.class_names}
        for cate in self.takeInCate:
            classCount[cate] = classCount.get(cate, 0) + 1
        for cate in self.takeOutCate:
            classCount[cate] = max([classCount.get(cate) - 1, 0])
        self.result = []
        self.result_dict = {}
        for key, value in classCount.items():
            if value != 0:
                self.result.append([key, value])
                self.result_dict.update({self.name2class[key]: value})

class Visulization:
    def __init__(self, font='SIMKAI.TTF', color=(0, 0, 0), thickness=1):
        self.font = font
        self.color = color
        self.thickness = thickness

    @staticmethod
    def drawCircle(self, img):

        pass

    @staticmethod
    def drawRectangle(self, draw, p):
        draw.rectangle(p, outline=self.color)  # p --> [x1, y1, x2, y2]

    @staticmethod
    def drawText(self):

        pass



