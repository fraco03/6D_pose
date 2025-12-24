# 6D pose estimation full pipeline


## Random samples from each 6D prediction model
Samples reported are taken using ground truth bounding boxes

<img src="README_imgs/sample/rgb.png" width="100%"/> <br/> <img src="README_imgs/sample/rgbd.png" width="100%"/> <br/> 
<img src="README_imgs/sample/pointnet.png" width="100%"/> <br/> <img src="README_imgs/sample/dense.png" width="100%"/>

## Evaluation benchmarks
We first tested the ADD metric using GT bboxes to evaluate the accuracy of the 6D pose prediction models, then we tested again using YOLO bounding boxes and finally compared the results.

We used 0.1D \% ADD metric

You can check the comparison steps [here](src/model_comparison.py).

You can check the evaluation steps [here](src/model_evaluation.py)
- Note: each one of the models has its own easy to use evaluation pipeline in `src/{model}/evaluation.py`, and can be seen in action at the end of training notebooks in `notebooks/{model}/{model}_training.ipynb`

### Using GT bboxes
<img src="README_imgs/gt_eval/accuracy.png" width="49%"/> <img src="README_imgs/gt_eval/mean_add.png" width="49%"/> <br/> 
<img src="README_imgs/gt_eval/per_object_accuracy.png" width="49%"/> <img src="README_imgs/gt_eval/per_object_add.png" width="49%"/>

### Using YOLO bboxes
<img src="README_imgs/yolo_eval/accuracy.png" width="49%"/> <img src="README_imgs/yolo_eval/mean_add.png" width="49%"/> <br/> 
<img src="README_imgs/yolo_eval/per_object_accuracy.png" width="49%"/> <img src="README_imgs/yolo_eval/per_object_add.png" width="49%"/>

### Comparison between GT and YOLO bounding boxes
<img src="README_imgs/comparison_eval/comparison.png" width="49%"/> <img src="README_imgs/comparison_eval/heatmap.png" width="49%"/>

## Full pipeline examples
In the full pipeline we don't have GTs anymore, so we perform pose estimation on each one of the detections provided by YOLO. 

Since the pose model has only been trained to predict pose for one object per image, we do not expect exceptional results for all the objects' pose prediction, especially for those who are occluded.

You can find the notebook using all the pipelines for all the models [here](inference/full_inference.ipynb)

<img src="README_imgs/pipeline/RGB.png" width="49%"/> <img src="README_imgs/pipeline/RGBD.png" width="49%"/> <br/>
<img src="README_imgs/pipeline/PointNet.png" width="49%"/> <img src="README_imgs/pipeline/DenseFusion.png" width="49%"/>
