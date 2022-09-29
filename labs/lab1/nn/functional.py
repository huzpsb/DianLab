from .modules import Module
import numpy as np
import math
import cupy as cp
bs = 128
# class Sigmoid(Module):
#
#     def forward(self, x):
#         # TODO Implement forward propogation
#         # of sigmoid function.
#
#         ...
#
#         # End of todo
#
#     def backward(self, dy):
#         # TODO Implement backward propogation
#         # of sigmoid function.
#
#         ...
#
#         # End of todo


def crosseloss(t,y):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta),axis = -1,keepdims = True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        # 下面用于保存梯度信息
        self.dW = None
        self.db = None

    def forward(self, x):
        # 用于保存原始输入
        self.x = x

        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class Tanh(Module):

    def forward(self, x):
        # TODO Implement forward propogation
        # of tanh function.

        ...

        # End of todo

    def backward(self, dy):
        # TODO Implement backward propogation
        # of tanh function.

        ...

        # End of todo

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
# class ReLU(Module):
#
#     def forward(self, x):
#         # TODO Implement forward propogation
#         # of ReLU function.
#
#         ...
#
#         # End of todo
#
#     def backward(self, dy):
#         # TODO Implement backward propogation
#         # of ReLU function.
#
#         ...
#
#         # End of todo


# class Softmax(Module):
#
#     def forward(self, x):
#         # TODO Implement forward propogation
#         # of Softmax function.
#
#         ...
#
#         # End of todo
#
#     def backward(self, dy):
#         # Omitted.
#         ...
def softmax(x):
    # #y = np.zeros_like(b)
    # row_max = cp.max(X, axis=1).reshape(-1, 1)
    # #print(row_max)
    # X -= row_max
    # X_exp = np.exp(X)
    # s = X_exp / cp.sum(X_exp, axis=1, keepdims=True)
    #print(np.sum(X_exp, axis=1, keepdims=True))
    x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    #return s


def crosseloss(t,y):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta),axis = -1,keepdims = True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        # if self.t.size == self.y.size:
        #     dx = (self.y - self.t) / batch_size
        # else:
        #     dx = self.y.copy()
        #     dx[np.arange(batch_size), self.t] -= 1
        #     dx = dx / batch_size

        return dx


# def Nu_Gr(f,l1,l2,l3,b1,b2):
#     h = 1e-4
#     gr1 = np.zeros_like(l1)
#     gr2 = np.zeros_like(l2)
#     gr3 = np.zeros_like(l3)
#     gb1 = np.zeros_like(b1)
#     gb2 = np.zeros_like(b2)
#     for idx in range(l1.size):
#         tmp_va = l1[idx]
#         l1[idx] = tmp_va + h
#         fxh1 = f(l1,l2,l3,b1,b2)
#         l1[idx] = tmp_va - h
#         fxh2 = f(l1,l2,l3,b1,b2)
#         gr1[idx] = (fxh1-fxh2)/(2*h)
#         l1[idx] = tmp_va
#     for idx in range(l2.size):
#         tmp_va = l2[idx]
#         l2[idx] = tmp_va + h
#         fxh1 = f(l1,l2,l3,b1,b2)
#         l2[idx] = tmp_va - h
#         fxh2 = f(l1,l2,l3,b1,b2)
#         gr2[idx] = (fxh1-fxh2)/(2*h)
#         l2[idx] = tmp_va
#     for idx in range(l3.size):
#         tmp_va = l3[idx]
#         l3[idx] = tmp_va + h
#         fxh1 = f(l1,l2,l3,b1,b2)
#         l3[idx] = tmp_va - h
#         fxh2 = f(l1,l2,l3,b1,b2)
#         gr3[idx] = (fxh1-fxh2)/(2*h)
#         l3[idx] = tmp_va
#     for idx in range(b1.size):
#         tmp_va = b1[idx]
#         b1[idx] = tmp_va + h
#         fxh1 = f(l1, l2, l3, b1, b2)
#         b1[idx] = tmp_va - h
#         fxh2 = f(l1, l2, l3, b1, b2)
#         gb1[idx] = (fxh1 - fxh2) / (2 * h)
#         b1[idx] = tmp_va
#     for idx in range(b2.size):
#         tmp_va = b2[idx]
#         b2[idx] = tmp_va + h
#         fxh1 = f(l1, l2, l3, b1, b2)
#         b2[idx] = tmp_va - h
#         fxh2 = f(l1, l2, l3, b1, b2)
#         gb2[idx] = (fxh1 - fxh2) / (2 * h)
#         b2[idx] = tmp_va
#     return gr1,gr2,gr3,gb1,gb2


# class Loss(object):
#     """
#     Usage:
#         >>> criterion = Loss(n_classes)
#         >>> ...
#         >>> for epoch in n_epochs:
#         ...     ...
#         ...     probs = model(x)
#         ...     loss = criterion(probs, target)
#         ...     model.backward(loss.backward())
#         ...     ...
#     """
#
#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#
#     def __call__(self, probs, targets):
#         self.probs = probs
#         self.targets = targets
#         ...
#         return self
#
#     def backward(self):
#         ...


# class SoftmaxLoss(Loss):
#
#     def __call__(self, probs, targets):
#         # TODO Calculate softmax loss.
#
#         ...
#
#         # End of todo
#
#     def backward(self):
#         # TODO Implement backward propogation
#         # of softmax loss function.
#
#         ...
#
#         # End of todo





# class CrossLoss():
#     def __init__(self,X,y):
#         self.X = X
#         self.y = y
#
#
#     def __call__(self,t, y):
#         #print("ty",t.size,y.size)
#         delta = 1e-7
#         return -np.sum(t* np.log(y + delta))

    # def Nu_Gr(self, f , l1, l2, l3, b1, b2):
    #     h = 1e-4
    #     gr1 = np.zeros_like(l1)
    #     gr2 = np.zeros_like(l2)
    #     gr3 = np.zeros_like(l3)
    #     gb1 = np.zeros_like(b1)
    #     gb2 = np.zeros_like(b2)
    #     print("X",l1.size)
    #     for idx in range(784):
    #         print(idx)
    #         for id in range(100):
    #
    #             #print(idx)
    #             tmp_va = l1[idx][id]
    #             l1[idx][id] = tmp_va + h
    #             fxh1 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #             l1[idx][id] = tmp_va - h
    #             fxh2 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #             gr1[idx][id] = (fxh1 - fxh2) / (2 * h)
    #             l1[idx][id] = tmp_va
    #     print("X", l2.size)
    #     for idx in range(100):
    #         for id in range(50):
    #             tmp_va = l2[idx][id]
    #             l2[idx][id] = tmp_va + h
    #             fxh1 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #             l2[idx][id] = tmp_va - h
    #             fxh2 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #             gr2[idx][id] = (fxh1 - fxh2) / (2 * h)
    #             l2[idx][id] = tmp_va
    #     print("X", l3.size)
    #     for idx in range(50):
    #         for id in range(10):
    #             tmp_va = l3[idx][id]
    #             l3[idx][id] = tmp_va + h
    #             fxh1 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #             l3[idx][id] = tmp_va - h
    #             fxh2 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #             gr3[idx][id] = (fxh1 - fxh2) / (2 * h)
    #             l3[idx][id] = tmp_va
    #     print("X", b1.size)
    #     for idx in range(b1.size):
    #         tmp_va = b1[idx]
    #         b1[idx] = tmp_va + h
    #         fxh1 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #         b1[idx] = tmp_va - h
    #         fxh2 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #         gb1[idx] = (fxh1 - fxh2) / (2 * h)
    #         b1[idx] = tmp_va
    #     print("X", b2.size)
    #     for idx in range(b2.size):
    #         tmp_va = b2[idx]
    #         b2[idx] = tmp_va + h
    #         fxh1 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #         b2[idx] = tmp_va - h
    #         fxh2 = self.__call__(f(self.X,l1, l2, l3, b1, b2,default = 1),self.y)
    #         gb2[idx] = (fxh1 - fxh2) / (2 * h)
    #         b2[idx] = tmp_va
    #     return gr1, gr2, gr3, gb1, gb2
   # def __call__(self, ):
        # TODO Calculate cross-entropy loss.
        #super().__call__()
       # ...
    # for col in range(targets.shape[-1]):
    #     probs[col] = probs[col] if probs[col] < 1 else 0.99999
    #     probs[col] = probs[col] if probs[col] > 0 else 0.00001
    #     C += targets[col] * np.log(probs[col]) + (1 - targets[col]) * np.log(1 - y_pred[col])
    # return -C



        # End of todo

   # def backward(self):
        # TODO Implement backward propogation
        # of cross-entropy loss function.

        ...

        # End of todo
