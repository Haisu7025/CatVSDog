# 清华大学 人工智能导论 第二次大作业 猫狗识别

## 文件清单：
* logs:日志文件夹
    * cross_val.log:10折交叉验证实验日志文件
    * resnet.log:使用resnet训练和验证的日志文件
    * resnet_pretrained.log:使用pytorch提供的预训练模型参数的resnet18网络训练和验证的日志文件
* src:代码源文件夹
    * cross_val_plot.py:交叉验证结果绘制文件夹
    * cross_val.py:交叉验证实验
    * extract_feature.py:特征提取实验
    * logger.py:日志文件配置初始化
    * resplot.py:resnet18训练验证结果绘制
    * train2.py:resnet18训练验证实验
    * visualize.py:resnet18网络结构可视化
* trained_models:训练好的模型文件夹
    * best_model.pth:resnet18训练验证过程的最佳模型
    * resnet18-5c106cde.pth:pytorch提供的预训练resnet18模型

## 实验环境
* python 2.7
* pytorch(numpy)
* graphviz
* matplotlib