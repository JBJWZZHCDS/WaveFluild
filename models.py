import torch
import torch.nn as nn
import torch.nn.functional as func
from params import params
from math import log2
from torch.nn.utils import weight_norm
from modules import DWT,RDWT,MultiPeriodSpeechAttention,MultiPeriodSpeechAttentionWithQ,SpeechUpsampleBlock,SpeechDownSampleBlock,SpeechResBlockDis,SpeechResBlock,SpeechResBlockRef

class DownSampleBlock(nn.Module):
    def __init__(self,inChannels,outChannels,patches,downSample,attMidChannels,attGroups,downGroups,resGroups,dropRate):
        super().__init__()
        self.attention=MultiPeriodSpeechAttention(outChannels,attMidChannels,attGroups,patches,dropRate)
        self.downSampler=SpeechDownSampleBlock(inChannels,outChannels,downGroups,downSample)
        self.resblocks=nn.Sequential(
            SpeechResBlockDis(outChannels,resGroups),
            SpeechResBlockDis(outChannels,resGroups),
            SpeechResBlockDis(outChannels,resGroups),
        )
        
    def setDropKey(self,value):
        self.attention.setDropKey(value)
            
    def forward(self,x):
        x=self.downSampler(x)
        x=self.attention(x)
        x=self.resblocks(x)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self,inChannels,outChannels,patches,upsample,attMidChannels,resMidChannels,attGroups,upGroups,resGroups,dropRate):
        super().__init__()
        self.attention=MultiPeriodSpeechAttentionWithQ(outChannels,attMidChannels,attGroups,patches,dropRate)
        self.upsampler=SpeechUpsampleBlock(inChannels,outChannels,upGroups,upsample)
        self.resblocks=nn.Sequential(
            SpeechResBlock(outChannels,resMidChannels,resGroups,1),
            SpeechResBlock(outChannels,resMidChannels,resGroups,3),
            SpeechResBlock(outChannels,resMidChannels,resGroups,5),
        )
    def setDropKey(self,value):
        self.attention.setDropKey(value)
        
    def forward(self,x,condition):
        x=self.upsampler(x)
        x=self.attention(x,condition)
        x=self.resblocks(x)
        return x
    
class UpSampleBlockSelf(nn.Module):
    def __init__(self,inChannels,outChannels,patches,upsample,attMidChannels,resMidChannels,attGroups,upGroups,resGroups,dropRate):
        super().__init__()
        self.attention=MultiPeriodSpeechAttention(outChannels,attMidChannels,attGroups,patches,dropRate)
        self.upsampler=SpeechUpsampleBlock(inChannels,outChannels,upGroups,upsample)
        self.resblocks=nn.Sequential(
            SpeechResBlock(outChannels,resMidChannels,resGroups,1),
            SpeechResBlock(outChannels,resMidChannels,resGroups,3),
            SpeechResBlock(outChannels,resMidChannels,resGroups,5),
        )
    def setDropKey(self,value):
        self.attention.setDropKey(value)
        
    def forward(self,x):
        x=self.upsampler(x)
        x=self.attention(x)
        x=self.resblocks(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,params=params):
        super().__init__()
        self.channels=params['channelsDis']
        self.attMidChannels=params['attMidChannelsDis']
        self.patches=params['patchesDis']
        self.attGroups=params['attGroupsDis']
        self.downSampleRates=params['downSampleRates']
        self.upGroups=params['downGroups']
        self.resGroups=params['resGroupsDis']
        self.dropRate=params['dropRateDis']
  
        self.melBands=params['melBands']
        self.melLen=params['melTrainWindow']
        self.audioLen=params['melTrainWindow']*(2**params['hopSize'])

        self.convIn=nn.Conv1d(1,self.channels[0],15,padding=7)
        self.attentionIn=MultiPeriodSpeechAttention(self.channels[0],self.attMidChannels[0],self.attGroups[0],self.patches[0],self.dropRate[0])
        self.resIn=nn.Sequential(
            SpeechResBlockDis(self.channels[0],self.resGroups[0]),
            SpeechResBlockDis(self.channels[0],self.resGroups[0]),
            SpeechResBlockDis(self.channels[0],self.resGroups[0]),
        )
        from math import log2
        self.dwts=nn.ModuleList()
        for i in range(len(self.downSampleRates)):
            self.dwts.append(DWT(round(log2(self.downSampleRates[i]))))
         
        self.convSkips=nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.convSkips.append(nn.Conv1d(self.downSampleRates[i]*self.channels[i],self.channels[i+1],1))
             
        self.blocks=nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(DownSampleBlock(self.channels[i],self.channels[i+1],self.patches[i+1],
                                             self.downSampleRates[i],self.attMidChannels[i+1],
                                             self.attGroups[i+1],self.upGroups[i],
                                             self.resGroups[i+1],self.dropRate[i+1]))
        self.act=nn.LeakyReLU(0.1)
        self.convOut1=nn.Conv1d(self.channels[-1],self.channels[-1],15,groups=self.resGroups[-1],padding='same')
        self.convOut2=nn.Conv1d(self.channels[-1],1,15,padding='same')
        
    def setDropKey(self,value):
        self.attentionIn.setDropKey(value)
        for i in range(len(self.blocks)):
            self.blocks[i].setDropKey(value)
        
    def forward(self,audio):
        
        audio=self.convIn(audio)
        audio=self.act(audio)
        audio=self.attentionIn(audio)
        audio=self.resIn(audio)
        
        downAudio=audio
        features=[audio]
        for i in range(len(self.channels)-1):
            if i==0:
                audio=self.blocks[i](audio)
            else:
                audio=self.blocks[i](audio+downAudio)
            downAudio=self.dwts[i](downAudio)
            downAudio=self.convSkips[i](downAudio)
            features.append(audio)
            
        value=self.convOut1(self.act(features[-1]))
        value=self.convOut2(self.act(value))
        return value,features
    
class Refiner(nn.Module):
    def __init__(self,params=params):
        super().__init__()
        self.downSampleRates=params['downSampleRatesRefiner']
        self.channels=params['channelsRefiner']
        self.groups=params['groupsRefiner']
        
        self.downs=nn.ModuleList()
        self.convIn1=weight_norm(nn.Conv1d(1,self.channels[0],15,padding='same'))
        self.convIn2=weight_norm(nn.Conv1d(self.channels[0],self.channels[0],15,padding='same'))
        self.convOut2=weight_norm(nn.Conv1d(self.channels[0],self.channels[0],15,padding='same'))
        self.convOut1=weight_norm(nn.Conv1d(self.channels[0],1,15,padding='same'))
        self.act=nn.LeakyReLU(0.1)
        
        for i in range(len(self.downSampleRates)):
            self.downs.append(SpeechDownSampleBlock(self.channels[i],self.channels[i+1],self.groups[i],self.downSampleRates[i]))
            
        self.downResBlocks=nn.ModuleList()
        for i in range(1,len(self.channels)):
            self.downResBlocks.append(nn.Sequential(
                SpeechResBlockRef(self.channels[i],self.groups[i-1]),
                SpeechResBlockRef(self.channels[i],self.groups[i-1]),
                SpeechResBlockRef(self.channels[i],self.groups[i-1]),
            )) 
            
        self.resMid=nn.Sequential(
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
            )
        
        self.dwtX=DWT(params['hopSize'])
        self.convX=weight_norm(nn.Conv1d(2**(params['hopSize']),self.channels[-1],7,padding='same'))
        self.resX=nn.Sequential(
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
            )
        self.timeIn1=weight_norm(nn.Linear(1,256))
        self.timeIn2=weight_norm(nn.Linear(256,self.channels[-1]))
        self.convMel=weight_norm(nn.Conv1d(params['melBands'],self.channels[-1],7,padding='same'))
        self.resMel=nn.Sequential(
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
                SpeechResBlockRef(self.channels[-1],self.groups[-1]),
            )
        
        self.upResBlocks=nn.ModuleList()
        self.convFuses=nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.upResBlocks.append(nn.Sequential(
                SpeechResBlock(self.channels[-i-1],self.channels[-i-1],self.groups[-i-1],1),
                SpeechResBlock(self.channels[-i-1],self.channels[-i-1],self.groups[-i-1],3),
                SpeechResBlock(self.channels[-i-1],self.channels[-i-1],self.groups[-i-1],5),
            )) 
            self.convFuses.append(nn.Conv1d(self.channels[-i-1]*2,self.channels[-i-1],1))
            
        self.ups=nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.ups.append(SpeechUpsampleBlock(self.channels[-i-1],self.channels[-i-2],self.groups[-i-1],self.downSampleRates[-i-1]))
        
   
    def forward(self,x,audio,time,mel):
        audio=self.act(self.convIn1(audio))
        audio=self.convIn2(audio)
        res=[]
        for i in range(len(self.downs)):
            audio=self.downs[i](audio)
            audio=self.downResBlocks[i](audio)
            res.append(audio)
            
        x=self.resX(self.act(self.convX(self.dwtX(x))))
        time=self.timeIn2(self.act(self.timeIn1(time)))
        mel=self.resMel(self.act(self.convMel(mel)))
        sum=x+mel+time.unsqueeze(-1)
       
        audio=self.resMid(audio)+sum
        
        for i in range(len(self.ups)):
            audio=self.convFuses[i](torch.cat((audio,res[-i-1]),dim=1))
            audio=self.upResBlocks[i](audio)
            audio=self.ups[i](audio)
        
        audio=self.act(self.convOut2(audio))
        audio=torch.tanh(self.convOut1(audio))
        return audio
            

class Generator(nn.Module):
    def __init__(self,params=params):
        super().__init__()
        self.channels=params['channelsGen']
        self.attMidChannels=params['attMidChannelsGen']
        self.patches=params['patchesGen']
        self.attGroups=params['attGroupsGen']
        self.upSampleRates=params['upSampleRates']
        self.upGroups=params['upGroups']
        self.resMidChannels=params['resMidChannelsGen']
        self.resGroups=params['resGroupsGen']
        self.dropRate=params['dropRateGen']
        
        self.melBands=params['melBands']
        self.melLen=params['melTrainWindow']
        self.audioLen=params['melTrainWindow']*(2**params['hopSize'])
        
        self.convIn=nn.Conv1d(self.melBands,self.channels[0],7,padding=3)
        self.attentionIn=MultiPeriodSpeechAttention(self.channels[0],self.attMidChannels[0],self.attGroups[0],self.patches[0],self.dropRate[0])
        self.resIn=nn.Sequential(
            SpeechResBlock(self.channels[0],self.resMidChannels[0],self.resGroups[0],dilation=1),
            SpeechResBlock(self.channels[0],self.resMidChannels[0],self.resGroups[0],dilation=3),
            SpeechResBlock(self.channels[0],self.resMidChannels[0],self.resGroups[0],dilation=5),
        )

         
        self.convSkips=nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.convSkips.append(nn.Conv1d(self.channels[i],self.channels[i+1],1))
             
        self.blocks=nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.blocks.append(UpSampleBlockSelf(self.channels[i],self.channels[i+1],self.patches[i+1],
                                             self.upSampleRates[i],self.attMidChannels[i+1],self.resMidChannels[i+1],
                                             self.attGroups[i+1],self.upGroups[i],self.resGroups[i+1],self.dropRate[i+1]))
        self.convOut1=nn.Conv1d(self.channels[-1],self.channels[-1],3,padding=1)
        self.convOut2=nn.Conv1d(self.channels[-1],1,3,padding=1)
        self.act=nn.LeakyReLU(0.1)
    def setDropKey(self,value):
        self.attentionIn.setDropKey(value)
        for i in range(len(self.blocks)):
            self.blocks[i].setDropKey(value)
            
    def forward(self,mel):
        
        mel=self.convIn(mel)
        mel=self.act(mel)
        mel=self.attentionIn(mel)
        mel=self.resIn(mel)
        
        res=mel
        for i in range(len(self.channels)-1):
            mel=self.blocks[i](mel)
            res=self.convSkips[i](res)
            res=func.interpolate(res,size=res.size(-1)*self.upSampleRates[i],mode='nearest')
            res=res+mel
           
        res=self.convOut1(self.act(res))
        res=self.convOut2(self.act(res))
        res=torch.tanh(res)
        return res
             

def generate(generator,refiner,mels,steps=None,fixFirst=False,scaler=None):
    if fixFirst==True:
        with torch.no_grad():
            audiosPre=generator(mels)
    else:
        audiosPre=generator(mels)
    
        
    if steps==None:
        return audiosPre
    else:
        x=torch.randn(mels.size(0),1,mels.size(-1)*(2**params['hopSize']),device=mels.device)
        time=torch.ones(mels.size(0),1,device=mels.device)
    
        for i in range(0,len(steps)):
            deltaTime=time*(1-steps[i])
            predicted=refiner(x,audiosPre,time,mels)
            velocity=(predicted-x)/time.unsqueeze(-1) 
            time=time-deltaTime
            if scaler!=None:
                velocity=velocity*scaler[i]
            x=x+velocity*deltaTime.unsqueeze(-1)
            
        predicted=refiner(x,audiosPre,time,mels)
        return (predicted+audiosPre)/2
           
class Scheduler(nn.Module):
    def __init__(self,step=params['schedulerStep']):
        super().__init__()
        self.scheduler=nn.Parameter(torch.rand(step-1)*(params['timeDecayUp']-params['timeDecayDown'])+params['timeDecayDown'])
        self.scaler=nn.Parameter(torch.ones(step-1))
    def clamp(self):
        self.scheduler.data.clamp_(0.025,1)
        self.scaler.data.clamp_(0.05,1.9)
            
    
    def getScheduler(self):
        return self.scheduler
    
    def getScaler(self):
        return self.scaler    
    
     
      
        
        
      
            


    
            
        
    
        