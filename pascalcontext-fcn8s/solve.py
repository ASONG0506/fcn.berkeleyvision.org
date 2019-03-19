#-*- coding:utf-8 -*-

# 进行训练的脚本文件
# 重要的是把网络的deconv层改成了双线性插值。就是调用的surgery函数的地方
import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../pascalcontext-fcn16s/pascalcontext-fcn16s.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
# 外部手术式修改网络的上采样的层
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/pascal/VOC2010/ImageSets/Main/val.txt', dtype=str)

# 训练和测试
for _ in range(50):
    solver.step(8000)
    score.seg_tests(solver, False, val, layer='score')
