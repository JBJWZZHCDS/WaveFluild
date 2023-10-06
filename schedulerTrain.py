import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as func
import torchaudio
import os
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import numpy as np  

from params import params
from dataset import AudioMelSet
from models import Generator,Discriminator,Scheduler,generate,Refiner,DWT
from params import params


    
def schedulerTrain(mixTraining=False):
    
    trainData=AudioMelSet(params['schedulerTrainDataPath'],params['schedulerTrainDevice'])
    g=Generator().to(params['schedulerTrainDevice'])
    r=Refiner().to(params['schedulerTrainDevice'])
    d=Discriminator().to(params['schedulerTrainDevice'])
    scheduler=Scheduler().to(params['schedulerTrainDevice'])
   
    schedulerLearnRate=params['schedulerLearnRate']

    g.load_state_dict(torch.load(params['generatorPathSch']))
    r.load_state_dict(torch.load(params['refinerPathSch']))
    d.load_state_dict(torch.load(params['discriminatorPathSch']))
    
    if os.path.exists(params['schedulerPath']):
        scheduler.load_state_dict(torch.load(params['schedulerPath']))
    else:
        torch.save(scheduler.state_dict(),params['schedulerPath'])

    schedulerOptimizer=optim.SGD(scheduler.parameters(),lr=schedulerLearnRate)

    trainLoader=torch.utils.data.DataLoader(
        trainData,batch_size=params['schedulerTrainBatch'],shuffle=True
    )
    
    epochs=params['schedulerEpoch']
    scaler=torch.cuda.amp.GradScaler(enabled=mixTraining)
    dwt=DWT(4)
    g.train()
    count=0
    for epoch in range(0,epochs):
        tqdmLoader=tqdm(trainLoader,desc=f'scheduler Train Epoch: {epoch}')
        g.setDropKey(False)
        d.setDropKey(False)
        
        for (audios,mels) in tqdmLoader:
            
            with torch.cuda.amp.autocast(enabled=mixTraining):
                fakes=generate(g,r,mels,scheduler.getScheduler(),True,scheduler.getScaler())
                    
                fakeValues,fakeFeatures=d(fakes)
                realFeatures=d(audios)[1]
                featureLoss=0

                for i in range(len(fakeFeatures)):
                    featureLoss+=params['featureScale']*(fakeFeatures[i]-realFeatures[i]).abs().mean()
                    
                fakeMels=trainData.getMel(fakes)
                melLoss=params['melScale']*(fakeMels-mels).abs().mean()
                # average L1 mel spectrogram loss
                    
                gMel=melLoss.item()/params['melScale']
                gFeature=featureLoss.item()/params['featureScale']
                gF=fakeValues.mean().item()
                
                dwtLoss=params['dwtScale']*(dwt(fakes)-dwt(audios)).abs().mean()
                gDwt=dwtLoss.item()/params['dwtScale']
                
                gLoss=2*((fakeValues-1)**2).mean()+melLoss+featureLoss+dwtLoss
                
            schedulerOptimizer.zero_grad()
            scaler.scale(gLoss).backward()
            scaler.unscale_(schedulerOptimizer)
            scaler.step(schedulerOptimizer)
            scaler.update()
                
            schedulerInfo=[round(one,2) for one in scheduler.getScheduler().detach().cpu().tolist()]
            scalerInfo=[round(one,2) for one in scheduler.getScaler().detach().cpu().tolist()]
            #print(scheduler.grad,scheduler.requires_grad)
            scheduler.clamp()
            tqdmLoader.set_postfix(gF=gF,gMel=gMel,gFeature=gFeature,gDwt=gDwt,scheduler=schedulerInfo,scaler=scalerInfo)


        torch.save(scheduler.state_dict(),params['schedulerPath'])


if __name__=='__main__':
    schedulerTrain()
    
