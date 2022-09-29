import numpy as np
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
#from numba import cuda
import cupy as cp
#print(cuda.gpus)
#cuda.select_device(0)
import nn
import nn.functional as F

n_features = 28 * 28
n_classes = 10
n_epochs = 10000
bs = 100
lr = 1e-3
FL = 50
SL = 20
In = 784
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
        h = 1e-8
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
        # for idx in range(In):
        #     if idx % 10 == 0:
        #         print(idx)
        #     for id in range(FL):
        #         # print(idx)
        #         tmp_va = self.L1[idx][id]
        #         self.L1[idx][id] = tmp_va + h
        #         fxh1 = self.loss(X,y)
        #         self.L1[idx][id] = tmp_va - h
        #         fxh2 = self.loss(X,y)
        #         gr1[idx][id] = (fxh1 - fxh2) / (2 * h)
        #         self.L1[idx][id] = tmp_va
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

    # End of todo


def load_mnist(mode='train', n_samples=None, flatten=True):
    images = './train-images.idx3-ubyte' if mode == 'train' else './t10k-images.idx3-ubyte'
    labels = './train-labels.idx1-ubyte' if mode == 'train' else './t10k-labels.idx1-ubyte'
    length = 60000 if mode == 'train' else 10000

    X = np.fromfile(open(images), np.uint8)[16:].reshape(
        (length, 28, 28)).astype(np.int32)
    if flatten:
        X = X.reshape(length, -1)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape(
        (length)).astype(np.int32)
    return (X[:n_samples] if n_samples is not None else X,
            y[:n_samples] if n_samples is not None else y)


def vis_demo(model):
    X, y = load_mnist('test', 20)
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    fig = plt.subplots(nrows=4, ncols=5, sharex='all',
                       sharey='all')[1].flatten()
    for i in range(20):
        img = X[i].reshape(28, 28)
        fig[i].set_title(preds[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.savefig("vis.png")
    plt.show()


def main():
    trainloader = nn.data.DataLoader(load_mnist('train'), batch=bs)
    testloader = nn.data.DataLoader(load_mnist('test'))
    #print(trainloader.shape())
    model = Model()
    optimizer = nn.optim.SGD(model, lr=lr, momentum=0.9)
    #criterion = F.CrossEntropyLoss(n_classes=n_classes)

    for i in range(n_epochs):
        bar = tqdm(trainloader, total=6e4 / bs)
        bar.set_description(f'epoch  {i:2}')
        for X, y in bar:
            X = X/255.0
            y1 = y
            #print(y.shape())
            z = np.zeros((y.size,10),)
            for j in range(y.size):
                z[j][y[j]] = 1
            #print(np.sum(X),z)
            y = z
            #print(y.shape())
            #print("XYSIZE",X.size,y.size)
            probs = model.forward(X)
            #print(probs.shape)
            #loss = F.CrossLoss(probs, y)
            #for k in range(10):
            model.backward(X,y)
            optimizer.Up_Wt()
        # gr1, gr2, gr3, gb1, gb2, gb3 = model.Nu_Gr(X, y)
        # print(gr2)

            probs = model.forward(X)
            preds = np.argmax(probs, axis=1)
            bar.set_postfix_str(f'acc={np.sum(preds == y1) / len(y) * 100:.1f}'
                                ' loss={loss.value:.3f}')

        for X, y in testloader:
            probs = model.forward(X)
            preds = np.argmax(probs, axis=1)
            print(f' test acc: {np.sum(preds == y) / len(y) * 100:.1f}')

    vis_demo(model)


if __name__ == '__main__':
    main()
