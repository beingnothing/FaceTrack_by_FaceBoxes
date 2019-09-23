# FaceBoxes-tensorflow

This repo is an implementation of Face Tracking based on [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234).  
I provide full training code, data preparation scripts, and a pretrained pb model.  


## About using the pretrained FaceBoxes detection model

To use the pretrained face detector you will need to download `face_detector.py` and  
a frozen inference graph (`.pb` file, it is [here](https://drive.google.com/drive/folders/1DYdxvMXm6n6BsOy4dOTbN9h43F0CoUoK?usp=sharing)). You can see an example of usage in `try_detector.ipynb`.

## Requirements

* tensorflow 1.10
* opencv-python, Pillow, tqdm

## Notes

1. *Warning:* This detector doesn't work well on small faces.  
But you can improve its performance if you upscale images before feeding them to the network.  
For example, resize an image keeping its aspect ratio so its smaller dimension is 768.
2. You can see how anchor densification works in `visualize_densified_anchor_boxes.ipynb`.
3. You can see how my data augmentation works in `test_input_pipeline.ipynb`.
4. The speed on a CPU is **~30 ms/image** (image size is 1024x768).
5. The detector has speed **~7 ms/image** (image size is 1024x1024, GPU is NVIDIA GeForce GTX 1080).

## About training FaceBoxes model

For training, use `train`+`val` parts of the WIDER dataset, it is 16106 images in total (12880 + 3226). 
For evaluation during the training, use the FDDB dataset (2845 images) and `AP@IOU=0.5` metrics (it is not like in the original FDDB evaluation, but like in PASCAL VOC Challenge).

1. Run `prepare_data/explore_and_prepare_WIDER.ipynb` to prepare the WIDER dataset 
(also, you will need to combine the two created dataset parts using `cp train_part2/* train/ -a`).
2. Run `prepare_data/explore_and_prepare_FDDB.ipynb` to prepare the FDDB dataset.
3. Create tfrecords:
  ```
  python create_tfrecords.py \
      --image_dir=/home/gpu2/hdd/dan/WIDER/train/images/ \
      --annotations_dir=/home/gpu2/hdd/dan/WIDER/train/annotations/ \
      --output=data/train_shards/ \
      --num_shards=150

  python create_tfrecords.py \
      --image_dir=/home/gpu2/hdd/dan/FDDB/val/images/ \
      --annotations_dir=/home/gpu2/hdd/dan/FDDB/val/annotations/ \
      --output=data/val_shards/ \
      --num_shards=20
  ```
4. Run `python train.py` to train a face detector. Evaluation on FDDB will happen periodically.
5. Run `tensorboard --logdir=models/run00` to observe training and evaluation.
6. Run `python save.py` and `create_pb.py` to convert the trained model into a `.pb` file.
7. Use `class` in `face_detector.py` and `.pb` file to do inference.
8. Also, you can get my final training checkpoint [here](https://drive.google.com/drive/folders/1DYdxvMXm6n6BsOy4dOTbN9h43F0CoUoK?usp=sharing).


## Evaluating on FDDB

1. Download the evaluation code from [here](http://vis-www.cs.umass.edu/fddb/results.html).
2. `tar -zxvf evaluation.tgz; cd evaluation`.  
Then compile it using `make` (it can be very tricky to make it work).
3. Run `predict_for_FDDB.ipynb` to make predictions on the evaluation dataset.  
You will get `ellipseList.txt`, `faceList.txt`, `detections.txt`, and `images/`.
4. Run `./evaluate -a result/ellipseList.txt -d result/detections.txt -i result/images/ -l result/faceList.txt -z .jpg -f 0`.
5. You will get something like `eval_results/discrete-ROC.txt`.
6. Run `eval_results/plot_roc.ipynb` to plot the curve.

Also see this [repository](https://github.com/pkdogcom/fddb-evaluate) and the official [FAQ](http://vis-www.cs.umass.edu/fddb/faq.html) if you have questions about the evaluation.

## Face tracking on video
I use kalman tracker for face tracking based on FaceBoxes detection model
To run the python code you have to put all the input videos under a folder like **./media** and then provide the path of that folder as command line argument:
```sh
python3 start.py ./media 
```
* Then you can find faces extracted saved in the folder **./facepics** .

## LICENSE
MIT LICENSE