# Cabinet Phase I
Automatic checkout system of the convenience cabinet by only visual signals.

## People
Project leaders: Jiangning Zhang  
Project verifiers:

## Requirements
- Ubuntu 16.04
- python 3.7
- cuda 10
- opencv
- pytorch 1.3.1
- mmcv
- [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md)

```angular2
pip install pytorch torchvision -c pytorch
pip install mmcv
cd mmdetection
python setup.py develop  # or "pip install -v -e ."
```

## Training 
- Download [dataset](https://pan.baidu.com/s/1ufZ2t5dDrt-EVGziWWqWog)`pw:e64b`
- Training with a signle GPU

```
cd mmdetection
python3 tools/train.py ${CONFIG_FILE}
e.g.:
python3 tools/train.py configs_zjn/cascade_rcnn_x101_64x4d_fpn_1x_zjn.py
```

## Testing

- Download pretrained [model](https://pan.baidu.com/s/1jGt54esoZ3ovsCkoqMU3qQ)`pw:avo4` to `cascade_rcnn_x101_64x4d_fpn_1x/model_0724.pth`
- Testing with a signle GPU
```
cd mmdetection
python3 tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
e.g.:
python3 tools/test.py configs_zjn/cascade_rcnn_x101_64x4d_fpn_1x_zjn.py work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_zjn/model_0724.pth --out out.pkl --show
```
using `--show` to visualize the results

## Deployment test
- Download pretrained [model](https://pan.baidu.com/s/1jGt54esoZ3ovsCkoqMU3qQ)`pw:avo4` to `cascade_rcnn_x101_64x4d_fpn_1x/model_0724.pth`
```shell
cd deployment
python3 run.py
```
## Use my own datasets

The simplest way is to convert your dataset to existing dataset formats (COCO or PASCAL VOC).
Taking `VOC` format as an example.
1. Create dataset: `cabinet/mmdetection/mmdet/datasets/voc_my.py`.
2. Register dataset: `cabinet/mmdetection/mmdet/datasets/__init__.py`.

## Use my own config
1. Copy `cabinet/mmdetection/configs/cascade_rcnn_x101_32x4d_fpn_1x.py` to `cabinet/mmdetection/configs_zjn/cascade_rcnn_x101_32x4d_fpn_1x_zjn.py`.
2. Modify variables: `num_classes`,`dataset_type`,`data_root`,`ann_file`,`img_prefix`.

## License

TBA

## Citation

TBA

