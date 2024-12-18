<div align="center">
 <h1 align="center">Quick Draw</h1>
</div>


[![GitHub license](https://img.shields.io/github/license/khoapdaio/quick-draw)](https://github.com/khoapdaio/quick-draw/blob/main/LICENSE)



## Introduction

Here is my python source code for QuickDraw - an online game developed by google. With my code, you could run an app which you could draw in front of a camera (If you use laptop, your webcam will be used by default) using hand detective

## Dataset
The dataset used for training my model could be found at [Quick Draw dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap). Here I only picked up 20 files for 20 categories


## Categories:
The table below shows 20 categories my model used:

| **#** | **Item**      | **#** | **Item**      | **#** | **Item**      | **#** | **Item**      |
|-------|---------------|-------|---------------|-------|---------------|-------|---------------|
| 1     | Apple         | 6     | Cup           | 11    | Hammer        | 16    | Star          |
| 2     | Book          | 7     | Door          | 12    | Hat           | 17    | T-shirt       |
| 3     | Bowtie        | 8     | Envelope      | 13    | Ice Cream     | 18    | Pants         |
| 4     | Candle        | 9     | Eyeglasses    | 14    | Leaf          | 19    | Lightning     |
| 5     | Cloud         | 10    | Guitar        | 15    | Scissors      | 20    | Tree          |

## Trained models

You could find my trained model at **`trained_models/improved_quickdraw_model.pth`**

## Training

You need to download npz files corresponding to 20 classes my model used and store them in folder **data**.
If you want to train your model with different list of categories, 
you only need to change the constant **CLASSES** at **src/config.py** and download necessary npz files.
Then you could simply run **python3 train.py**


