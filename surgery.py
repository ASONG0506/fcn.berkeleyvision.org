#-* -coding:utf-8 -*-

#进行网络权重的迁移
#非常感谢这篇博客的介绍：https://blog.csdn.net/qq_21368481/article/details/80289350

from __future__ import division
import caffe
import numpy as np

def transplant(new_net, net, suffix=''):
    """
    新旧网络符合的参数直接进行复制；
    新旧网络的参数的维度不同的参数，超过新的的维度范围的参数直接舍弃
    新旧参数不匹配的，由于网络的参数数量还是相同的，所以进行强制转换。也就是用flat的参数直接复制

    根据打印出来的信息，所有的参数中前面基层2 2 3 3 3的参数都是直接复制，而之前的fc层参数前置转化进行了flat之后进行的复制，就是flat的参数复制过去以后，
    接收参数的权重的形状还是不会发生变化，是采用卷积核的形式进行表示的。都是通过Python的flat函数得以实现。
    这里的一个关键就是参数的数量要一直，所以在fcn中，fc6卷积核是7，fc7卷积核是1的卷积核就是为了保证参数数量相同的。


    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.

    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.

    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    """
    for p in net.params:
        p_new = p + suffix
        # 如果新的没有这些参数，直接舍弃
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        # 如果参数形状不匹配或者符合
        # caffe中的参数的net.param[][0]表示权重参数，而net.param[][1]表示偏置bias
        for i in range(len(net.params[p])):
            # 超过维度的直接舍弃
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            # 参数的维度不符合，打印出来correct参数 的信息；参数符合，打印复制的信息。归根到底，因为形状都一样，所以都是拉平了直接复制的
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            # 参数拉平直接复制
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

def upsample_filt(size):
    """
    为了生成双线性插值的核参数
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    # 返回的是size*size大小的核
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp(net, layers):
    """
    调用前面的函数进行反卷积的权重初始化
    Set weights of each layer in layers to bilinear kernels for interpolation.
    """
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

def expand_score(new_net, new_layer, net, layer):
    """
    扩展score层，将旧网络中的参数移到新的网络中，新的网络的class比旧的要小
    Transplant an old score layer's parameters, with k < k' classes, into a new
    score layer with k classes s.t. the first k' are the old classes.
    """
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data
