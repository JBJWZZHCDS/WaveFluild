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
from models import Generator,Refiner,Discriminator,generate
from params import params
from modules import DWT

def train(mixTraining=False):
    trainData=AudioMelSet(params['trainDataPath'],params['trainDevice'])
    g=Generator().to(params['trainDevice'])
    r=Refiner().to(params['trainDevice'])
    d=Discriminator().to(params['trainDevice'])
    learnRateG=params['learnRateG']
    learnRateD=params['learnRateD']
    learnRateR=params['learnRateR']
    betas=params['betas']
    weightDecay=params['weightDecay']

    if os.path.exists(params['generatorPath']):
        g.load_state_dict(torch.load(params['generatorPath']))
    else:
        torch.save(g.state_dict(),params['generatorPath'])
    
    if os.path.exists(params['refinerPath']):
        r.load_state_dict(torch.load(params['refinerPath']))
    else:
        torch.save(r.state_dict(),params['refinerPath'])
                   
    if os.path.exists(params['discriminatorPath']):
        d.load_state_dict(torch.load(params['discriminatorPath']))
    else:
        torch.save(d.state_dict(),params['discriminatorPath'])  
        
    gOptimizer=optim.AdamW(g.parameters(),lr=learnRateG,betas=betas,weight_decay=weightDecay)
    rOptimizer=optim.AdamW(r.parameters(),lr=learnRateR,betas=betas,weight_decay=weightDecay)
    dOptimizer=optim.AdamW(d.parameters(),lr=learnRateD,betas=betas,weight_decay=weightDecay)
    gScheduler=optim.lr_scheduler.ExponentialLR(gOptimizer,gamma=params['gamma'])
    rScheduler=optim.lr_scheduler.ExponentialLR(rOptimizer,gamma=params['gamma'])
    dScheduler=optim.lr_scheduler.ExponentialLR(dOptimizer,gamma=params['gamma'])
    trainLoader=torch.utils.data.DataLoader(
        trainData,batch_size=params['trainBatch'],shuffle=True
    )

    epochs=params['trainEpoch']
    scaler=torch.cuda.amp.GradScaler(enabled=mixTraining)
    dwt=DWT(4)
    g.train()
    for epoch in range(0,epochs):
        tqdmLoader=tqdm(trainLoader,desc=f'train Epoch: {epoch}')
        if epoch>=params['dropKeyEndEpoch']:
            g.setDropKey(False)
            d.setDropKey(False)
        else:
            g.setDropKey(True)
            d.setDropKey(True)

        for (audios,mels) in tqdmLoader:
                
            with torch.cuda.amp.autocast(enabled=mixTraining):
                scheduler=torch.rand(torch.randint(0,params['timeStep'],size=(1,)))*(params['timeDecayUp']-params['timeDecayDown'])+params['timeDecayDown']
                fakes=generate(g,r,mels,scheduler,False)
                    
                fakeValues=d(fakes.detach())[0]
                realValues=d(audios)[0]
                
                dF=fakeValues.mean().item()
                dR=realValues.mean().item()

                dLoss=(fakeValues**2).mean()+((realValues-1)**2).mean()
                dOptimizer.zero_grad()
                scaler.scale(dLoss).backward()
                scaler.unscale_(dOptimizer)
                scaler.step(dOptimizer)
                scaler.update()
                    

                fakeValues,fakeFeatures=d(fakes)
                _,realFeatures=d(audios)
                
                featureLoss=0
                for i in range(len(fakeFeatures)):
                    featureLoss+=params['featureScale']*(fakeFeatures[i]-realFeatures[i]).abs().mean() 
                        
                fakeMels=trainData.getMel(fakes)

                melLoss=params['melScale']*(fakeMels-mels).abs().mean()
                dwtLoss=params['dwtScale']*(dwt(fakes)-dwt(audios)).abs().mean()
                gDwt=dwtLoss.item()/params['dwtScale']
                gMel=melLoss.item()/params['melScale']
                gFeature=featureLoss.item()/params['featureScale']
                gF=fakeValues.mean().item()
                gLoss=2*((fakeValues-1)**2).mean()+dwtLoss+melLoss+featureLoss
                
                gOptimizer.zero_grad()
                rOptimizer.zero_grad()
                scaler.scale(gLoss).backward()
                scaler.unscale_(gOptimizer)
                scaler.unscale_(rOptimizer)
                scaler.step(gOptimizer)
                scaler.step(rOptimizer)
                scaler.update()
            
            
            tqdmLoader.set_postfix(dF=dF,dR=dR,gF=gF,gMel=gMel,gFeature=gFeature,gDwt=gDwt)   
            #tqdmLoader.set_postfix(dF=dF,dR=dR,dN=dN,dL=dL,gF=gF,gMel=gMel,gFeature=gFeature)
        torch.save(g.state_dict(),params['generatorPath'])
        torch.save(r.state_dict(),params['refinerPath'])
        torch.save(d.state_dict(),params['discriminatorPath'])
        gScheduler.step()
        dScheduler.step()
        rScheduler.step()
       


if __name__=='__main__':
    train()
    
