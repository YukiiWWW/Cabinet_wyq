# from mmdet.apis import init_detector, inference_detector

# config_file = './mmdetection/configs_sxt/my_config_ssd300.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
# # url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = './mmdetection/checkpoints/ssd_epoch_24.pth'
# device = 'cpu'
# # init a detector
# model = init_detector(config_file, checkpoint_file, device=device)
# # inference the demo image
# img = './mmdetection/test.jpg'
# result = inference_detector(model, img)
# model.show_result(img, result, out_file='result2.jpg')

status,a = (True, (463, 492, 115, 116))
xhyh = list(a)
print(xhyh)
import cv2

