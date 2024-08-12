import copy
from typing import Optional
import torch


def set_requires_grad(net, requires_grad=False):
    """
    Set requires_grad=False for all the parameters to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


class EMATeacher2(object):

    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        self.teacher = copy.deepcopy(model)
        set_requires_grad(self.teacher, False)

    def set_alpha(self, alpha: float):
        assert alpha >= 0
        self.alpha = alpha

    def update(self):
        for teacher_param, param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data = self.alpha * teacher_param + (1 - self.alpha) * param

    def __call__(self, x: torch.Tensor):
        return self.teacher(x)

    def train(self, mode: Optional[bool] = True):
        self.teacher.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.teacher.state_dict()

    def load_state_dict(self, state_dict):
        self.teacher.load_state_dict(state_dict)

    @property
    def module(self):
        return self.teacher.module

class EMATeacher(torch.nn.Module):

    def __init__(self, model, alpha):
        super(EMATeacher, self).__init__()  
        self.model = model
        self.alpha = alpha
        self.teacher = copy.deepcopy(model)
        # set_requires_grad(self.teacher, False)

    def set_alpha(self, alpha: float):
        assert alpha >= 0
        self.alpha = alpha

    def update(self):
        
        for teacher_param, param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data = self.alpha * teacher_param + (1 - self.alpha) * param

    def __call__(self, x: torch.Tensor):
        return self.teacher(x)

    def train(self, mode: Optional[bool] = True):
        self.teacher.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.teacher.state_dict()

    def load_state_dict(self, state_dict):
        self.teacher.load_state_dict(state_dict)

    @property
    def module(self):
        return self.teacher.module
    
def update_bn(model, ema_model):
    """
    Replace batch normalization statistics of the teacher model with that ot the student model
    """
    for m2, m1 in zip(model.named_modules(), ema_model.named_modules()):
        if ('bn' in m2[0]) and ('bn' in m1[0]):
            bn2, bn1 = m2[1].state_dict(), m1[1].state_dict()
            bn2['running_mean'].data.copy_(bn1['running_mean'].data)
            bn2['running_var'].data.copy_(bn1['running_var'].data)
            bn2['num_batches_tracked'].data.copy_(bn1['num_batches_tracked'].data)
