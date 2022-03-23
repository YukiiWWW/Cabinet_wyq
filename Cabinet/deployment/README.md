# Cabinet Phase I

Automatic checkout system of the convenience cabinet by only visual signals.
## Requirements
- Ubuntu 16.04
- python3.7.3
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
- Train with a signle GPU

```
python3 tools/train.py ${CONFIG_FILE}
e.g.:
python3 tools/train.py configs_zjn/cascade_rcnn_x101_64x4d_fpn_1x_zjn.py
```

- Train with a multiple GPUs

```
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
e.g.:
./tools/dist_train.sh configs_zjn/cascade_rcnn_x101_64x4d_fpn_1x_zjn.py 2
```
## Testing
```
# single-gpu testing
python3 tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

## Testing video
```shell
   python3 test_video.py
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

