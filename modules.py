from typing import Iterable, Optional
import torch
from torch.nn.modules.module import Module
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as func
import torchaudio


class DWT(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple=multiple
        self.length=2**multiple

    def forward(self,x):      
        assert len(x.shape)==3,"In DWT layer,input tensor should be in N,C,T format"
        batch,channel,timestep = x.shape
        if(timestep%self.length!=0):
            x=func.pad(x,[0,self.length-timestep%self.length],mode='constant',value=0)
            timestep=x.size(-1)
       
        for i in range(0,self.multiple):
            channel=channel*2
            timestep=timestep//2

            x_next=torch.zeros(batch,channel,timestep,device=x.device)
            x_next[:,0::2,:]=(x[:,:,0::2]+x[:,:,1::2])/torch.sqrt(torch.tensor(2))# Low resolution
            x_next[:,1::2,:]=(x[:,:,0::2]-x[:,:,1::2])/torch.sqrt(torch.tensor(2))# High resolution
            x=x_next
            
            if i==self.multiple-1:
                return x
            
        return None
    
class RDWT(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple=multiple
        self.length=2**multiple

    def forward(self,x):      
        assert len(x.shape)==3,"In RDWT layer,input tensor should be in N,C,T format"
        batch,channel,timestep = x.shape

        for i in range(0,self.multiple):
            channel=channel//2
            timestep=timestep*2
           
            x_next=torch.zeros(batch,channel,timestep,device=x.device)
            x_next[:,:,0::2]=(x[:,0::2,:]+x[:,1::2,:])/torch.sqrt(torch.tensor(2))# Low resolution
            x_next[:,:,1::2]=(x[:,0::2,:]-x[:,1::2,:])/torch.sqrt(torch.tensor(2))# High resolution
            x=x_next
            
            if i==self.multiple-1:
                return x
            
        return None

from torch.nn.utils import weight_norm,remove_weight_norm
class SpeechResBlock(nn.Module):
    def __init__(self,channels,midChannels,groups,dilation=1):
        super().__init__()
        self.slope=0.1
        self.act=nn.LeakyReLU(self.slope)
        self.convs=nn.ModuleList([
            weight_norm(nn.Conv1d(channels,midChannels,3,dilation=1,groups=groups,padding='same')),
            weight_norm(nn.Conv1d(midChannels,channels,3,dilation=dilation,groups=groups,padding='same')),
            weight_norm(nn.Conv1d(channels,midChannels,1,dilation=1,padding='same')),
            weight_norm(nn.Conv1d(midChannels,channels,5,dilation=dilation,groups=groups,padding='same')),
            weight_norm(nn.Conv1d(channels,midChannels,1,dilation=1,padding='same')),
            weight_norm(nn.Conv1d(midChannels,channels,7,dilation=dilation,groups=groups,padding='same')),  
        ])
                
    def forward(self,x):
        x3=self.convs[1](self.act(self.convs[0](self.act(x))))
        x5=self.convs[3](self.act(self.convs[2](self.act(x))))
        x7=self.convs[5](self.act(self.convs[4](self.act(x))))
        return x+(x3+x5+x7)/torch.sqrt(torch.tensor(3))

class SpeechResBlockDis(nn.Module):
    def __init__(self,channels,groups):
        super().__init__()
        self.slope=0.1
        self.act=nn.LeakyReLU(self.slope)
        self.convs=nn.ModuleList([
            weight_norm(nn.Conv1d(channels,channels,37,groups=groups,padding='same')),
            weight_norm(nn.Conv1d(channels,channels,41,groups=groups,padding='same')),
            weight_norm(nn.Conv1d(channels,channels,47,groups=groups,padding='same')),
        ])
                
    def forward(self,x):
        x37=self.convs[0](self.act(x))
        x41=self.convs[1](self.act(x))
        x47=self.convs[2](self.act(x))
        return x+(x37+x41+x47)/torch.sqrt(torch.tensor(3))

class SpeechResBlockRef(nn.Module):
    def __init__(self,channels,groups):
        super().__init__()
        self.slope=0.1
        self.act=nn.LeakyReLU(self.slope)
        self.convs=nn.ModuleList([
            weight_norm(nn.Conv1d(channels,channels,11,groups=groups,padding='same')),
            weight_norm(nn.Conv1d(channels,channels,13,groups=groups,padding='same')),
            weight_norm(nn.Conv1d(channels,channels,17,groups=groups,padding='same')),
        ])
                
    def forward(self,x):
        x11=self.convs[0](self.act(x))
        x13=self.convs[1](self.act(x))
        x17=self.convs[2](self.act(x))
        return x+(x11+x13+x17)/torch.sqrt(torch.tensor(3))
     
class SpeechDownSampleBlock(nn.Module):
    def __init__(self,inChannels,outChannels,groups,downSample=8):
        super().__init__()
        self.slope=0.1
        self.downSample=downSample
        self.act=nn.LeakyReLU(self.slope)
        self.convs=nn.ModuleList([
            weight_norm(nn.Conv1d(inChannels,outChannels,4*downSample+1,stride=downSample,groups=groups,padding=2*downSample)),
            weight_norm(nn.Conv1d(inChannels,outChannels,6*downSample+1,stride=downSample,groups=groups,padding=3*downSample)),     
        ])
        self.convSkip=weight_norm(nn.Conv1d(inChannels,outChannels,1))
        
    def forward(self,x):
        x1=func.interpolate(self.convSkip(x),size=x.size(-1)//self.downSample,mode='nearest')
        x3=self.convs[0](self.act(x))
        x5=self.convs[1](self.act(x))
        return x1+(x3+x5)/torch.sqrt(torch.tensor(2))
    
class SpeechUpsampleBlock(nn.Module):
    def __init__(self,inChannels,outChannels,groups,upSample=8):
        super().__init__()
        self.slope=0.1
        self.upSample=upSample
        self.act=nn.LeakyReLU(self.slope)
        self.convs=nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(inChannels,outChannels,3*upSample,stride=upSample,groups=groups,padding=upSample)),
            weight_norm(nn.Conv1d(outChannels,outChannels,3,padding='same')),
            weight_norm(nn.ConvTranspose1d(inChannels,outChannels,5*upSample,stride=upSample,groups=groups,padding=2*upSample)),
            weight_norm(nn.Conv1d(outChannels,outChannels,3,padding='same')),       
        ])
        self.convSkip=weight_norm(nn.Conv1d(inChannels,outChannels,1))
        
               
    def forward(self,x):
        x1=func.interpolate(self.convSkip(x),size=x.size(-1)*self.upSample,mode='nearest')
        x3=self.convs[1](self.act(self.convs[0](self.act(x))))
        x5=self.convs[3](self.act(self.convs[2](self.act(x))))
        return x1+(x3+x5)/torch.sqrt(torch.tensor(2))

class SpeechAttention(nn.Module):
    def __init__(self,channels,midChannels,groups,patch,dropRate=0.1):
        super().__init__()
        self.channels,self.midChannels,self.groups=channels,midChannels,groups
        self.patch,self.dropRate=patch,dropRate

        self.convQKV=weight_norm(nn.Conv2d(channels,3*midChannels,(5,1),groups=groups,padding='same'))   
        self.convOut=weight_norm(nn.Conv1d(midChannels,channels,3,padding='same'))
        
        self.drop=True
        
        
    def forward(self,x):
        
        padding=0 if x.size(-1)%self.patch==0 else self.patch-x.size(-1)%self.patch
        x=func.pad(x,(0,padding),mode='constant',value=0)

        x=x.reshape(x.size(0),self.channels,x.size(-1)//self.patch,self.patch)
        
        q,k,v=self.convQKV(x).chunk(chunks=3,dim=1)
        if self.drop==True:
            mask=torch.bernoulli(k,self.dropRate)*(-1e9)
            k=k+mask.detach()
        
        batch,channel,wordNum,patch=x.size(0),self.midChannels,x.size(-2),self.patch

        q=q.transpose(1,2).reshape(batch,wordNum,channel*patch)
        k=k.transpose(1,2).reshape(batch,wordNum,channel*patch)
        v=v.transpose(1,2).reshape(batch,wordNum,channel*patch)
        
        q=q.softmax(dim=-1)
        k=k.softmax(dim=-2)
        out=torch.einsum('Bik,Bkj->Bij',q,torch.einsum('Bki,Bkj->Bij',k,v))
        out=out.reshape(batch,wordNum,channel,patch).transpose(1,2).reshape(batch,channel,wordNum*patch)
        out=self.convOut(out)
        if padding!=0:
            out=out[:,:,:-padding]
        return out

class MultiPeriodSpeechAttention(nn.Module):
    def __init__(self,channels,midChannels,groups,patches,dropRate=0.1):
        super().__init__()
        self.attentions=nn.ModuleList()
        for patch in patches:
            self.attentions.append(SpeechAttention(channels,midChannels,groups,patch,dropRate))
    
    def setDropKey(self,value):
        for i in range(len(self.attentions)):
            self.attentions[i].drop=value 
    
    def forward(self,x):
        out=None
        for layer in self.attentions:
            if out==None:
                out=layer(x)
            else:
                out=out+layer(x)
        return x+out/torch.sqrt(torch.tensor(len(self.attentions)))
                
class SpeechAttentionWithQ(nn.Module):
    def __init__(self,channels,midChannels,groups,patch,dropRate=0.1):
        super().__init__()
        self.channels,self.midChannels,self.groups=channels,midChannels,groups
        self.patch,self.dropRate=patch,dropRate

        self.convQ=weight_norm(nn.Conv2d(channels,midChannels,(5,1),groups=groups,padding='same'))
        self.convKV=weight_norm(nn.Conv2d(channels,2*midChannels,(5,1),groups=groups,padding='same'))   
        self.convOut=weight_norm(nn.Conv1d(midChannels,channels,3,padding='same'))
        
        self.drop=True
        
        
    def forward(self,x,condition):
        
        paddingX=0 if x.size(-1)%self.patch==0 else self.patch-x.size(-1)%self.patch
        paddingCondition=0 if condition.size(-1)%self.patch==0 else self.patch-condition.size(-1)%self.patch
        
        x=func.pad(x,(0,paddingX),mode='constant',value=0)
        x=x.reshape(x.size(0),self.channels,x.size(-1)//self.patch,self.patch)
        condition=func.pad(condition,(0,paddingCondition),mode='constant',value=0)
        condition=condition.reshape(condition.size(0),self.channels,condition.size(-1)//self.patch,self.patch)
        
        q=self.convQ(condition)
        k,v=self.convKV(x).chunk(chunks=2,dim=1)
        if self.drop==True:
            mask=torch.bernoulli(k,self.dropRate)*(-1e9)
            k=k+mask.detach()
        
        batch,channel,wordNum,patch=x.size(0),self.midChannels,x.size(-2),self.patch
        q=q.transpose(1,2).reshape(batch,wordNum,channel*patch)
        k=k.transpose(1,2).reshape(batch,wordNum,channel*patch)
        v=v.transpose(1,2).reshape(batch,wordNum,channel*patch)
        
        q=q.softmax(dim=-1)
        k=k.softmax(dim=-2)
        out=torch.einsum('Bik,Bkj->Bij',q,torch.einsum('Bki,Bkj->Bij',k,v))
        out=out.reshape(batch,wordNum,channel,patch).transpose(1,2).reshape(batch,channel,wordNum*patch)
        out=self.convOut(out)
        if paddingX!=0:
            out=out[:,:,:-paddingX]
        return out
    
class MultiPeriodSpeechAttentionWithQ(nn.Module):
    def __init__(self,channels,midChannels,groups,patches,dropRate=0.1):
        super().__init__()
        self.attentions=nn.ModuleList()
        
        for patch in patches:
            self.attentions.append(SpeechAttentionWithQ(channels,midChannels,groups,patch,dropRate))
    
    def setDropKey(self,value):
        for i in range(len(self.attentions)):
            self.attentionsv[i].drop=value
        
    
    def forward(self,x,condition):
        out=None
        for layer in self.attentions:
            if out==None:
                out=layer(x,condition)
            else:
                out=out+layer(x,condition)
        return x+condition+out/torch.sqrt(torch.tensor(len(self.attentions)))