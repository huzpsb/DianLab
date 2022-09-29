class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):
        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.

        if module == 'Tensor':
            self._update_weight()
        # elif module == 'Module':
        #     self._step_module()
        return
        # End of todo

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD():

    def __init__(self, target, lr, momentum: float = 0):
        #super(SGD, self).__init__(module, lr)
        self.model = target
        self.lr = lr
        self.momentum = momentum

    def Up_Wt(self,):
        # TODO Update the weight of tensor
        # in SGD manner.
        self.model.L1 -= self.model.Gr1 * self.lr
        self.model.L2 -= self.model.Gr2 * self.lr
        self.model.L3 -= self.model.Gr3 * self.lr
        self.model.B1 -= self.model.Gb1 * self.lr
        self.model.B2 -= self.model.Gb2 * self.lr
        self.model.B3 -= self.model.Gb3 * self.lr
        return
        ...

        # End of todo


class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)

        # TODO Initialize the attributes
        # of Adam optimizer.

        ...

        # End of todo

    def _update_weight(self, tensor):
        # TODO Update the weight of
        # tensor in Adam manner.

        ...

        # End of todo
