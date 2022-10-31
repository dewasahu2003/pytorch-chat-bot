import torch.nn as nn

class Bot_Model(nn.Module):

    def __init__(self,input_size,hidden_size,num_classes):
        super().__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.l3=nn.Linear(hidden_size,num_classes)
        self.relu=nn.ReLU()

    def forward(self,x):
        keep_going=self.relu(self.l1(x))
        keep_going=self.relu(self.l2(keep_going))
        keep_going=self.l3(keep_going)
        return keep_going



