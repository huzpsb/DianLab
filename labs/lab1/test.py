import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import nn
import nn.functional as F
def f(x,y,z = 0):
    a = np.square(x) + np.square(y)
    b = np.sum(a)
    return b
n_features = 28 * 28
n_classes = 10
n_epochs = 10
bs = 1000
lr = 1e-3
In = 10
FL = 10
SL = 16
lengths = (n_features, 512, n_classes)

class Model(nn.Module):
    # TODO Design the classifier.
    def __init__(self):
        super().__init__()
        self.L1 = np.random.randn(In, FL)
        self.L2 = np.random.randn(FL, SL)
        self.L3 = np.random.randn(SL, 10)
        # self.L1 = np.ones((In, FL), dtype=float)
        # self.L2 = np.zeros((FL, SL), dtype=float)
        # self.L3 = np.zeros((SL, 10), dtype=float)
        self.B1 = np.zeros((FL,), dtype=float)
        self.B2 = np.zeros((SL,), dtype=float)
        self.B3 = np.zeros((n_classes,), dtype=float)
        self.Gr1 = np.zeros((In, FL), dtype=float)
        self.Gr2 = np.zeros((FL, SL), dtype=float)
        self.Gr3 = np.zeros((SL, 10), dtype=float)
        self.Gb1 = np.zeros((FL,), dtype=float)
        self.Gb2 = np.zeros((SL,), dtype=float)

        self.fc1 = F.Affine(self.L1, self.B1)
        self.fc2 = F.Sigmoid()
        self.fc3 = F.Affine(self.L2, self.B2)
        self.fc4 = F.Sigmoid()
        self.fc5 = F.Affine(self.L3, self.B3)
        self.fc55 = F.softmax
        self.fc6 = F.SoftmaxWithLoss()

        self.Loss = None

    def load(self, X, y):
        self.X = X
        self.y = y

    # @cuda.jit
    def forward(self, X):
        # ,l1 = np.zeros((784,FL),dtype=float),l2 = np.zeros((FL, SL),dtype=float ),l3 = np.zeros((SL, 10),dtype=float ),\
        #       b1 = np.zeros((FL,),dtype=float),b2 = np.zeros((SL,),dtype=float),default = 0):
        # if not default:
        # l1 = self.L1
        # l2 = self.L2
        # l3 = self.L3
        # b1 = self.B1
        # b2 = self.B2
        # print(X,l1)

        # out = cp.dot(X,l1)
        # #start = time.time()
        # out = F.Sigmoid(out+b1)
        # #end = time.time()
        # out = cp.dot(out,l2)
        # out = F.Sigmoid(out+b2)
        # out = cp.dot(out,l3)
        # out = F.softmax(out)
        out = self.fc1.forward(X)
        out = self.fc2.forward(out)
        out = self.fc3.forward(out)
        out = self.fc4.forward(out)
        out = self.fc5.forward(out)
        self.fc5out = out
        out = self.fc55(out)



        # print(1000*(end-start))
        return out

    def loss(self,X,y):
        out = self.fc1.forward(X)
        out = self.fc2.forward(out)
        out = self.fc3.forward(out)
        out = self.fc4.forward(out)
        out = self.fc5.forward(out)
        #out = X + self.B1
        out = self.fc6.forward(out,y)
        return out

    def backward(self, X, y):
        self.loss(X,y)
        dout = self.fc6.backward()
        dout = self.fc5.backward(dout)
        dout = self.fc4.backward(dout)
        dout = self.fc3.backward(dout)
        dout = self.fc2.backward(dout)
        dout = self.fc1.backward(dout)
        self.Gr1 = self.fc1.dW
        self.Gb1 = self.fc1.db
        self.Gr2 = self.fc3.dW
        self.Gb2 = self.fc3.db
        self.Gr3 = self.fc5.dW
        self.Gb3 = self.fc5.db
        #dout = self.fc6
        # cr = self.load(X,y)
        # f = self.forward
        # print("back")
        # self.Gr1,self.Gr2,self.Gr3,self.Gb1,self.Gb2 = self.Nu_Gr(f,self.L1,self.L2,self.L3,self.B1,self.B2)
        return

    def CrossLoss(self, t, y):
        # print("ty",t.size,y.size)
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

    def Nu_Gr(self, X,y):
        h = 1e-6
        l1 = self.L1
        l2 = self.L2
        l3 = self.L3
        b1 = self.B1
        b2 = self.B2
        b3 = self.B3
        gr1 = np.zeros_like(l1)
        gr2 = np.zeros_like(l2)
        gr3 = np.zeros_like(l3)
        gb1 = np.zeros_like(b1)
        gb2 = np.zeros_like(b2)
        gb3 = np.zeros_like(b3)
        print("X", l1.size)
        f = self.forward
        for idx in range(In):
            if idx % 100 == 0:
                print(idx)
            for id in range(FL):
                # print(idx)
                tmp_va = self.L1[idx][id]
                self.L1[idx][id] = tmp_va + h
                fxh1 = self.loss(X,y)
                self.L1[idx][id] = tmp_va - h
                fxh2 = self.loss(X,y)
                gr1[idx][id] = (fxh1 - fxh2) / (2 * h)
                self.L1[idx][id] = tmp_va
        print("X", l2.size)
        for idx in range(FL):
            for id in range(SL):
                tmp_va = self.L2[idx][id]
                self.L2[idx][id] = tmp_va + h
                fxh1 = self.loss(X,y)
                self.L2[idx][id] = tmp_va - h
                fxh2 = self.loss(X,y)
                gr2[idx][id] = (fxh1 - fxh2) / (2 * h)
                self.L2[idx][id] = tmp_va
        print("X", l3.size)
        for idx in range(SL):
            for id in range(10):
                tmp_va = self.L3[idx][id]
                self.L3[idx][id] = tmp_va + h
                fxh1 = self.loss(X,y)
                self.L3[idx][id] = tmp_va - h
                fxh2 = self.loss(X,y)
                gr3[idx][id] = (fxh1 - fxh2) / (2 * h)
                self.L3[idx][id] = tmp_va
        print("X", b1.size)
        for idx in range(b1.size):
            tmp_va = self.B1[idx]
            self.B1[idx] = tmp_va + h
            fxh1 = self.loss(X,y)
            self.B1[idx] = tmp_va - h
            fxh2 = self.loss(X,y)
            gb1[idx] = (fxh1 - fxh2) / (2 * h)
            self.B1[idx] = tmp_va
        print("X", b2.size)
        for idx in range(b2.size):
            tmp_va = self.B2[idx]
            self.B2[idx] = tmp_va + h
            fxh1 = self.loss(X,y)
            self.B2[idx] = tmp_va - h
            fxh2 = self.loss(X,y)
            gb2[idx] = (fxh1 - fxh2) / (2 * h)
            self.B2[idx] = tmp_va
        for idx in range(b3.size):
            tmp_va = self.B3[idx]
            self.B3[idx] = tmp_va + h
            fxh1 = self.loss(X,y)
            self.B3[idx] = tmp_va - h
            fxh2 = self.loss(X,y)
            gb3[idx] = (fxh1 - fxh2) / (2 * h)
            self.B3[idx] = tmp_va
        return gr1, gr2, gr3, gb1, gb2,gb3


    ...

    ...

m = Model()
a = np.ones((1,In),dtype = float)
b = np.array([[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],])
print(m.forward(a))
m.load(a,b)
gr1 ,gr2,gr3,gb1,gb2,gb3= m.Nu_Gr(a,b)
m.backward(a,b)
op = nn.optim.SGD(m, lr=1e-3, momentum=0.9)
# for i in range(1000):
#
#     op.Up_Wt()
#     print(m.loss(a,b))
#     print(np.argmax(m.forward(a), axis=1))

print("here",m.backward(a,b),m.Gr1 - gr1,m.Gr2 - gr2,m.Gr3 - gr3,m.Gb1 - m.Gb1,gb1,m.Gb2 - gb2,m.Gb3 - gb3)
optimizer = nn.optim.SGD(m, lr=1e-3, momentum=0.9)
optimizer.Up_Wt()
print(m.forward(a))
#print(np.random.randn(784,100))
#print(np.zeros((4,10),))