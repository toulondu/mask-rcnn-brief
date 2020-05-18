## Mask R-CNN
Mask R-CNN来自何恺明大神2017年的论文，是一个通用的目标检测和实例分割的模型。它基于作者团队在2015年提出的faster rcnn模型，最主要的改动就是增加了一个分支来用于分割任务。
Mask R-CNN是anchor-based的模型，依然采用Faster RCNN的2-stage结构，首先用RPN找出候选region，然后在此基础上计算ROI并完成分类、检测和分割任务。
并没有添加各种trick，Mask RCNN就超过了当时所有的sota模型。

## repo介绍
本repo中代码使用pytorch modelzoo提供的现成Mask R-CNN预训练模型来进行fine-turing，实现一个目标检测和语义分割应用。并且在这个过程中来重新复习一下Mask R-CNN这个经典网络的一些原理。

介绍文章见[个人博客](https://toulondu.github.io/2020/05/19/%E8%BE%B9%E5%86%99%E4%BB%A3%E7%A0%81%E8%BE%B9%E5%AD%A6%E4%B9%A0mask-rcnn/)
