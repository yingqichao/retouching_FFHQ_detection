# RetouchingFFHQ-A-Large-scale-Dataset-for-Fine-grained-Face-Retouching-Detection
MM23: RetouchingFFHQ: A Large-scale Dataset for Fine-grained Face Retouching Detection

Hopefully this dataset & work would benefit you.

Paper: https://dl.acm.org/doi/pdf/10.1145/3581783.3611843

Dataset request: https://fdmas.github.io/Application_RetouchingFFHQ_new.pdf

checkpoints: https://drive.google.com/drive/folders/1bovqkik33bmwxtubtxlt6zljwpbogaqw?usp=sharing
(the files are renamed upon upload. please refer to meiyan_ckpt2yml.txt)

Should you encounter problem, contact me via shinydotcom@163.com, or open issues. Thanks!

为了方便阅读，本说明文件尽量用中文来表示
## 文件架构介绍
- data: 这里存放的是数据集的读取方法以及dataloader
- (+)model：这里存放的是模型脚本，其中base_model.py是基类，定义了一些基础方法，比如对图像进行攻击、模型保存等等。meiyan_baseline.py是对基类方法的继承，定义具体实验的train/test函数等。networks.py定义了常用网络架构
- noise_layers：定义了一些图像后处理攻击，例如blur、JPEG压缩等等
- (+)options：这里存放的是配置文件和配置文件如何读取，options.py是定义怎么读取对应的yml
- utils：其他的一些杂七杂八的帮助函数，比如创建文件夹之类的
- (+)sh: 运行的脚本
- (+)train.py，运行的主入口

## 控制台输出
控制台会收集模型train/test过程中的变量并自动求平均，把需要统计的变量（meiyan_baseline.py的Line 82/83塞了acc和loss）放进logs的字典里返回即可

求平均的逻辑在 Line 110-130

以下是控制台的样例
```
[1, 5000 79968 1 0.0001] lr: 0.0001 epoch_accuracy: 0.1296 loss: 2.3517 time per sample 0.0040 s
[1, 5000 79968 0 0.0001] lr: 0.0001 epoch_accuracy: 0.1291 loss: 2.3527 time per sample 0.0040 s
Saving models and training states.
Model saved to: /groupshare/meiyan_detection_results/model/ViT/4999_ViT.pth
[1, 5040 80608 1 0.0001] lr: 0.0001 epoch_accuracy: 0.1296 loss: 2.3515 time per sample 0.0040 s
[1, 5040 80608 0 0.0001] lr: 0.0001 epoch_accuracy: 0.1292 loss: 2.3525 time per sample 0.0040 s
```

## 模型与结果保存路径
模型和图像结果没有保存在项目文件夹下面，这个很重要，因为整个文件夹Git了，如果存在里面的话会导致Git的东西很多
已经设置好路径，为```/groupshare/meiyan_detection_results```，这个的设置在 Line 61, base_model.py (self.out_space_storage)
在sh里面可以指定```task_name```，这样的话不同实验可以有单独的二级目录，比如如果```task_name="ViT"```，则模型会在```/groupshare/meiyan_detection_results/ViT/models```下面


## 工作流
当你调用了sh文件（例如run_detection.sh）后：
```
python -m torch.distributed.launch --master_port 3111 --nproc_per_node=2 train.py \
          -opt options/meiyan_hallucinate.yml -mode 0 --launcher pytorch
```
- 参数： -opt会指定配置文件使用哪个，-task_name指定本次实验的名字（上面说过了，用来存模型指定路径等等），-nproc_per_node是枝使用几张卡，这个要记得和yml里面的gpu_ids（例如meiyan_hallucinate.yml的Line 7）保持一致
- 数据集加载： train.py Line 88 （这一行上面的都是定义args和启动分布式训练的语句，不用怎么修改）
- 模型定义： train.py Line 92
- 训练/测试脚本：train.py Line 98
- 标准训练测试脚本分为两个步骤：数据读入（例：feed_data_router）以及执行（例：train_ViT）
