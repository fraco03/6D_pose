# 6D pose estimation full pipeline


## Random samples from each 6D prediction model
Samples reported are taken using ground truth bounding boxes

<img src="README_imgs/sample/rgb.png" width="100%"/> <br/> <img src="README_imgs/sample/rgbd.png" width="100%"/> <br/> 
<img src="README_imgs/sample/pointnet.png" width="100%"/> <br/> <img src="README_imgs/sample/dense.png" width="100%"/>

## Evaluation benchmarks
We first tested the ADD metric using GT bboxes to evaluate the accuracy of the 6D pose prediction models, then we tested again using YOLO bounding boxes and finally compared the results.

We used 0.1D \% ADD metric

### Using GT bboxes
<img src="README_imgs/gt_eval/accuracy.png" width="49%"/> <img src="README_imgs/gt_eval/mean_add.png" width="49%"/> <br/> 
<img src="README_imgs/gt_eval/per_object_accuracy.png" width="49%"/> <img src="README_imgs/gt_eval/per_object_add.png" width="49%"/>

### Using YOLO bboxes
<img src="README_imgs/yolo_eval/accuracy.png" width="49%"/> <img src="README_imgs/yolo_eval/mean_add.png" width="49%"/> <br/> 
<img src="README_imgs/yolo_eval/per_object_accuracy.png" width="49%"/> <img src="README_imgs/yolo_eval/per_object_add.png" width="49%"/>

### Comparison between GT and YOLO bounding boxes
<img src="README_imgs/comparison_eval/comparison.png" width="49%"/> <img src="README_imgs/comparison_eval/heatmap.png" width="49%"/>