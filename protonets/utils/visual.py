from visdom import Visdom
import numpy as np

viz = Visdom()

def train_val_acc(layout):
    """

    """
    win = create
    def update_train_val_acc(epoch, trainAcc, valAcc):
        update(win, 'train', epoch, trainAcc)
        update(win, 'val', epoch, valAcc)
    return update_train_val_acc

def train_val_loss(epoch, trainLoss, valLoss, layout):
    """
    
    """
    pass

def create():
    win = viz
    return win

def update(win, line, x, y):
    pass

# line updates
from time import sleep
win = viz.line(
    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                        np.linspace(5, 10, 10) + 5)),
)
sleep(1)
viz.line(
    X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                        np.linspace(5, 10, 10) + 5)),
    win=win,
    update='append'
)
sleep(1)
viz.line(
    X=np.arange(21, 30),
    Y=np.arange(1, 10),
    win=win,
    name='2',
    update='append'
)
sleep(1)
viz.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='delete this',
    update='append'
)
sleep(1)
viz.line(
    X=np.arange(1, 10),
    Y=np.arange(11, 20),
    win=win,
    name='4',
    update='insert'
)
sleep(1)
viz.line(X=None, Y=None, win=win, name='delete this', update='remove')
input('Waiting for callbacks')