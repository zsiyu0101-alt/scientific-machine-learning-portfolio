import torch
import torch.nn as nn
from time import perf_counter
import matplotlib.pyplot as plt
import gc
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,sampler
from sys import stdout

class PendantGRU(nn.Module):
    def __init__(self,n_stack=4,h_dim=64,dropout=0.2):
        super(PendantGRU,self).__init__()
        self.rnn=nn.GRU(2,h_dim,n_stack,batch_first=True,dropout=dropout,bidirectional=True).cuda()
        self.linear=nn.Sequential(nn.Linear(2*h_dim,16),nn.ReLU(),nn.Linear(16,1)).cuda()
        self.relu=nn.ReLU()
        
    def forward(self,x):
        out=self.rnn(x)[0][::,-1,::]
        out=self.relu(out)
        out=self.linear(out)
        return out
    
    def parametersRNN(self):
        for m in self.rnn.parameters():
            yield m
    
    def parametersLinear(self):
        for m in self.linear.parameters():
            yield m
    
class PendantInception(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PendantInception,self).__init__()
        if not out_channels%4==0:
            Exception('out_channels not supported')
        oc=out_channels//4
        self.conv3=nn.Conv2d(in_channels,oc,(3,1),padding='same',dtype=torch.float32)
        self.conv5=nn.Conv2d(in_channels,oc,(5,1),padding='same',dtype=torch.float32)
#         self.conv5=nn.Conv2d(in_channels,oc,(1,5),padding='same',dtype=torch.float32)
#         self.conv7=nn.Conv2d(in_channels,oc,(1,7),padding='same',dtype=torch.float32)
        
        self.conv32=nn.Conv2d(in_channels,oc*2,(3,2),padding=(1,0),dtype=torch.float32)
        self.conv52=nn.Conv2d(in_channels,oc*2,(5,2),padding=(2,0),dtype=torch.float32)
        
#        self.relu=nn.ReLU()
        
        self.modulelist=[self.conv3,self.conv5,
                         self.conv32,self.conv52,
                        # self.relu
                        ]

    def forward(self,x):
        out=[]
        for m in self.modulelist:
            out.append(m(x))
        y1=torch.cat(out[:2],dim=1)
        y2=torch.cat(out[2:],dim=3)
#         for y in [y1,y2,y3]:
#             print(y.shape)
        return torch.cat([y1,y2],dim=1)


class PendantNet(nn.Module):
    def __init__(self,beta,R0):
        super(PendantNet,self).__init__()
        self.beta=beta
        self.R0=R0
        
    def forward(self,x):
        pred_R0=self.R0(x).squeeze()
        x_=x.squeeze()/pred_R0.reshape(-1,1,1)
        pred_beta=self.beta(x_).squeeze()
        return pred_beta,pred_R0
    
    def parametersBeta(self):
        for p in self.beta.parameters():
            yield p
    
    def parametersR0(self):
        for p in self.R0.parameters():
            yield p
    
    def train_(self,dataset,opt,scheduler=None,epochs=1,resample_every=5,batch_size=256,
          sample_parameters={'n':500,'mode':'random','pixel_quantize':True,'noise_level':1e-3}):
        self.train()
        switch=True
        N=len(dataset)
        for e in range(epochs):
            start=perf_counter()
            if (e%resample_every==0):
                gc.collect()
                torch.cuda.empty_cache()
                if switch:
                    sample_parameters['mode']='uniform'
                else:
                    sample_parameters['mode']='random'
                switch=not switch
                dataset.sample(**sample_parameters)
                loader=DataLoader(dataset,batch_size=batch_size,sampler=sampler.RandomSampler(range(len(dataset))))
            
            l1=0
            l2=0
            for x,y in loader:
                beta,R0=y.transpose(0,1)
                pred_beta,pred_R0=self.forward(x)
                loss2=F.l1_loss(pred_R0,R0,reduction='sum')
                loss1=F.l1_loss(pred_beta,beta,reduction='sum')                
                l1+=loss1.item()
                l2+=loss2.item()
                opt.zero_grad()
                loss2.backward(retain_graph=True)
                loss1.backward()
                opt.step()
            l1/=N
            l2/=N
            if not scheduler is None:
                scheduler.step()

            end=perf_counter()
            e_=e+1
            minu=int(end-start)//60
            sec=int(end-start)%60

            stdout.write('\r'+' '*30)
            stdout.write('\r epoch ({}/{})'.format(e_,epochs)+' took %d m %d s'%(minu,sec)+' loss = %.4f %.1f'%(l1,l2),)
            stdout.flush()