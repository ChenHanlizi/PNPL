<details>
<summary>简体中文</summary>

## 介绍

本项目是:page_with_curl:Learning Positive-Negative Prompts for Open-Set Remote Sensing Scene Classification的实现，如果您喜欢我们的项目，请点一下star:star:吧！

![75239387479](C:\Users\ADMINI~1\AppData\Local\Temp\1752393874794.png)

------

## 准备工作

:one:按照如下步骤安装本项目运行所需要的环境

`conda create -n NegPrompt python=3.8`
`conda activate NegPrompt`
`pip install -r requirements.txt`

:two:新建model_cache文件夹并将[clip的模型](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K)下载至此

:three:按照data文件夹中每个数据集的train和test的csv文件划分数据集，你可以直接下载[已经划分好的数据](https://pan.baidu.com/s/1KhOG-GULKrugVTSmGYDnOg?pwd=6666)并将其放置在对应的数据集文件夹下

对每个数据集的划分说明如下：

| dataset | unknown class num | unknown classes                                              |
| ------- | ----------------- | ------------------------------------------------------------ |
| UCM_p1  | 3                 | agricultural, beach, storagetanks                            |
| UCM_p1  | 5                 | agricultural, beach, chaparral, forest, river                |
| UCM_p1  | 9                 | baseball diamond, buildings, dense residential, golfcourse, medium residential, mobile homepark, parkinglot, sparse residential, tenniscourt |
| AID_p2  | 4                 | bareland, beach, desert, mountain                            |
| AID_p1  | 9                 | airport, bridge, church, parking, port, railway station, resort, storagetanks, viaduct |
| AID_p5  | 14                | airport, bareland, beach, bridge, church, desert, meadow, mountain, parking, port, railway station, resort, storage tanks, viaduct |
| NWPU_p1 | 6                 | beach, harbor, island, lake, river, sea ice                  |
| NWPU_p2 | 11                | airplane, bridge, church, freeway, intersection, overpass, palace, railway, railway station, roundabout, runway |
| NWPU_p3 | 15                | airplane, bridge, church, freeway, harbor, intersection, island, lake, overpass, railway, railway station, river, roundabout, runway, sea ice |

------

## 训练&测试

如果你需要重新训练

:heavy_exclamation_mark:注意修改参数：train_test_openset.py中，dataset为所需要运行的数据集，stage为不同阶段，1代表训练正面提示，3代表训练负面提示，6代表测试。与此同时，你需要修改datasets/osr_dataloader.py中对应数据集的函数的json和npy的名称。

| dataset | json                            | npy                            |
| ------- | ------------------------------- | ------------------------------ |
| UCM_p1  | ucm_class_index_unknown3.json   | ucm_unknown3_class_clean.npy   |
| UCM_p2  | ucm_class_index_unknown5.json   | ucm_unknown5_class_clean.npy   |
| UCM_p3  | ucm_class_index_unknown9.json   | ucm_unknown9_class_clean.npy   |
| AID_p2  | aid_class_index_unknown4.json   | aid_unknown4_class_clean.npy   |
| AID_p1  | aid_class_index_unknown9.json   | aid_unknown9_class_clean.npy   |
| AID_p5  | aid_class_index_unknown14.json  | aid_unknown14_class_clean.npy  |
| NWPU_p1 | nwpu_class_index_unknown6.json  | nwpu_unknown6_class_clean.npy  |
| NWPU_p2 | nwpu_class_index_unknown11.json | nwpu_unknown11_class_clean.npy |
| NWPU_p3 | nwpu_class_index_unknown15.json | nwpu_unknown15_class_clean.npy |

执行`python scripts/train_test_openset.py`

如果你只是想复现出和论文中一模一样的结果

请下载[我们的pth](https://pan.baidu.com/s/1BuOwmJhU9QWpI86WfdwzrQ?pwd=6666)，并直接运行测试代码(stage=6)，三个seed的平均值即为论文中的结果。



我们的代码参考了https://github.com/mala-lab/NegPrompt，感谢他们出色的工作。

</details>

<details>
<summary>English</summary>

## Introduction

This project implements :page_with_curl: Learning Positive-Negative Prompts for Open-Set Remote Sensing Scene Classification. If you like our project, please give us a star:star:!

![75239387479](C:\Users\ADMINI~1\AppData\Local\Temp\1752393874794.png)

------

## Preparation

:one: Set up the required environment with the following commands:

`conda create -n NegPrompt python=3.8`
`conda activate NegPrompt`
`pip install -r requirements.txt`

:two: Create a model_cache folder and download the [CLIP model](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K) into it

:three: Prepare datasets according to the train/test CSV files in the data folder for each dataset. You can directly download [pre-split data](https://pan.baidu.com/s/1KhOG-GULKrugVTSmGYDnOg?pwd=6666) and place it in the corresponding dataset folder.

Dataset splits are as follows:

| dataset | unknown class num | unknown classes                                              |
| ------- | ----------------- | ------------------------------------------------------------ |
| UCM_p1  | 3                 | agricultural, beach, storagetanks                            |
| UCM_p1  | 5                 | agricultural, beach, chaparral, forest, river                |
| UCM_p1  | 9                 | baseball diamond, buildings, dense residential, golfcourse, medium residential, mobile homepark, parkinglot, sparse residential, tenniscourt |
| AID_p2  | 4                 | bareland, beach, desert, mountain                            |
| AID_p1  | 9                 | airport, bridge, church, parking, port, railway station, resort, storagetanks, viaduct |
| AID_p5  | 14                | airport, bareland, beach, bridge, church, desert, meadow, mountain, parking, port, railway station, resort, storage tanks, viaduct |
| NWPU_p1 | 6                 | beach, harbor, island, lake, river, sea ice                  |
| NWPU_p2 | 11                | airplane, bridge, church, freeway, intersection, overpass, palace, railway, railway station, roundabout, runway |
| NWPU_p3 | 15                | airplane, bridge, church, freeway, harbor, intersection, island, lake, overpass, railway, railway station, river, roundabout, runway, sea ice |

------

## Training & Testing

If you need to retrain:

:heavy_exclamation_mark: Note to modify parameters: In train_test_openset.py, set 'dataset' to your target dataset, and 'stage' to different phases (1 for positive prompt training, 3 for negative prompt training, 6 for testing). Also modify the json and npy filenames in datasets/osr_dataloader.py for corresponding datasets.

| dataset | json                            | npy                            |
| ------- | ------------------------------- | ------------------------------ |
| UCM_p1  | ucm_class_index_unknown3.json   | ucm_unknown3_class_clean.npy   |
| UCM_p2  | ucm_class_index_unknown5.json   | ucm_unknown5_class_clean.npy   |
| UCM_p3  | ucm_class_index_unknown9.json   | ucm_unknown9_class_clean.npy   |
| AID_p2  | aid_class_index_unknown4.json   | aid_unknown4_class_clean.npy   |
| AID_p1  | aid_class_index_unknown9.json   | aid_unknown9_class_clean.npy   |
| AID_p5  | aid_class_index_unknown14.json  | aid_unknown14_class_clean.npy  |
| NWPU_p1 | nwpu_class_index_unknown6.json  | nwpu_unknown6_class_clean.npy  |
| NWPU_p2 | nwpu_class_index_unknown11.json | nwpu_unknown11_class_clean.npy |
| NWPU_p3 | nwpu_class_index_unknown15.json | nwpu_unknown15_class_clean.npy |

Run `python scripts/train_test_openset.py`

If you just want to reproduce the exact results from our paper:

Please download [our pth files](https://pan.baidu.com/s/1BuOwmJhU9QWpI86WfdwzrQ?pwd=6666) and directly run the testing code (stage=6). The average of three seeds will match the results in our paper.

Our code references https://github.com/mala-lab/NegPrompt, thanks for their excellent work.
</details>