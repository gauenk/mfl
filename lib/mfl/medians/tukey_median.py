import torch as th

class TukeyMedian():
    def __init__(self):
        self.a = 1

    def __call__(self,grads):
        return th.median(grads,1)
