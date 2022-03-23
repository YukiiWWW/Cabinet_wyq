from utils import *
import trajectory as tra
import os
import json
import pickle
import mmcv
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result
from datetime import datetime


class CabinetDetector:
    def __init__(self):
        self.config_file = 'configs_zjn/cascade_rcnn_x101_64x4d_fpn_1x_zjn.py'
        self.checkpoint_file = 'cascade_rcnn_x101_64x4d_fpn_1x/epoch_17.pth'
        self.model = init_detector(self.config_file, self.checkpoint_file, device='cuda:0')
        self.class_names = self.model.CLASSES
        # fw = open('class_names.txt', 'rb')
        # class_names = pickle.load(fw)
        self.img_show_size = (800, 450)
        self.lineProportion = [(0.00, 0.25), (1.00, 0.25)]

    def run(self, video_path, logpath, verbose=False, logging=False):
        f = open(logpath, 'w')
        matcher = tra.Matcher(self.lineProportion, 17, 5, self.class_names)
        # test a video and show the results
        video = mmcv.VideoReader(video_path)
        frameCount = 0
        f_open = False
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('demo/{}.avi'.format(os.path.basename(video_path).split('.')[0]), fourcc, 30, self.img_show_size)
        if len(video) > 800:
            return False, list(), dict()
        for frame in video:
            H, W, C = frame.shape
            # open/close
            meanRBG = frame[0:int(H * 0.2), int(0.2 * W):int(0.8 * W), :].mean()
            f_open = False
            if meanRBG > 60:
                f_open = True
            img_show = cv2.resize(frame, self.img_show_size)
            log_cur_frame = ''
            if f_open:
                result = inference_detector(self.model, frame)
                bboxes = np.vstack(result)
                labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
                labels = np.concatenate(labels)
                score_thr = 0.3
                if score_thr > 0:
                    assert bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > score_thr
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]

                bboxes[:, 0] /= W
                bboxes[:, 2] /= W
                bboxes[:, 1] /= H
                bboxes[:, 3] /= H
                bboxes = np.hstack([bboxes[:, 0:2], (bboxes[:, 2] - bboxes[:, 0])[:, np.newaxis],
                                    (bboxes[:, 3] - bboxes[:, 1])[:, np.newaxis], labels[:, np.newaxis]])
                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    log_cur_frame += '[{}: {:0.3f} {:0.3f} {:0.3f} {:0.3f}] '.format(int(bbox[4]), bbox[0]*W, bbox[1]*H, bbox[2]*W, bbox[3]*H)
                matcher.update(frame, bboxes)
                if verbose:
                    # judge and visualization
                    for trajectory in matcher.trajectories:
                        points = [(int((p[0] + p[2] / 2) * self.img_show_size[0]), int((p[1] + p[3] / 2) * self.img_show_size[1]))
                                  for p in trajectory.points]
                        color = trajectory.color
                        for p in points:
                            img_show = cv2.circle(img_show, p, 1, color, 2)
                    # cv2.line(img_show, (
                    # int(lineProportion[0][0] * img_show_size[0]), int(lineProportion[0][1] * img_show_size[1])),
                    #          (int(lineProportion[1][0] * img_show_size[0]),
                    #           int(lineProportion[1][1] * img_show_size[1])),
                    #          (255, 255, 255), 1)
                    img_show = draw_bboxes(img_show, bboxes, class_names=self.class_names)
                    texts = list()
                    texts.append([0.9, 0.1, 'Open', (255, 0, 0)])
                    texts.append([0.05, 0.1, 'frame  {}'.format(frameCount), (255, 0, 0)])
                    texts.append([0.05, 0.2, 'take   {}'.format(matcher.takeIn), (255, 0, 0)])
                    texts.append([0.05, 0.3, 'put    {}'.format(matcher.takeOut), (255, 0, 0)])
                    texts.append([0.05, 0.4, 'result {}'.format(matcher.result), (255, 0, 0)])

                    img_show = draw_text(img_show, texts)
            else:
                if verbose:
                    texts = list()
                    texts.append([0.9, 0.1, 'Close', (0, 0, 255)])
                    texts.append([0.05, 0.1, 'frame {}'.format(frameCount), (255, 0, 0)])
                    img_show = draw_text(img_show, texts)
            if verbose:
                cv2.imshow('show', img_show)
                cv2.waitKey(1)
            frameCount += 1
            if logging:
                print('\r{}/{}'.format(frameCount, len(video)), end='')
            log_string = ''
            log_string += '{}\t'.format(datetime.now().isoformat())
            log_string += '{}/{}\t'.format(frameCount, len(video))
            log_string += 'state: {}\t'.format('open' if f_open else 'close')
            log_string += 'take: {}\t'.format(matcher.takeIn)
            log_string += 'put: {}\t'.format(matcher.takeOut)
            log_string += 'updated result: {}\t'.format(matcher.result)
            log_string += 'detection results: {}\n'.format(log_cur_frame)
            f.write(log_string)
            # print(log_string)
            out.write(img_show)
        out.release()
        f.close()
        result = matcher.result
        result_dict = matcher.result_dict
        return True, result, result_dict


if __name__ == '__main__':
    test_single_video = True
    if test_single_video:
        video_path = 'video/0-0.mp4'
        logpath = 'logs/{}.txt'.format(os.path.basename(video_path))
        CD = CabinetDetector()
        state, result, result_dict = CD.run(video_path, logpath, verbose=True, logging=True)
        print(result)
    else:
        video_root = 'video'
        names = os.listdir(video_root)
        for name in names:
            video_path = os.path.join(video_root, name)
            logpath = 'logs/{}.txt'.format(os.path.basename(video_path))
            CD = CabinetDetector()
            state, result, result_dict = CD.run(video_path, logpath, verbose=True, logging=True)
            print(result)
