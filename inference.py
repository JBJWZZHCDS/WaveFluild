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
from models import Generator,Refiner,Scheduler,generate
from params import params
import torch
import torch.nn.functional as func
import torchaudio
from tqdm import tqdm
from params import params
import os


def inference():
        
    with torch.no_grad():
        melBands = params['melBands']
        fftSize = 2** params['fftSize']
        windowSize = 2** params['windowSize']
        hopSize = 2** params['hopSize']
        fmin = params['fmin']
        fmax= params['fmax']
        audios = []
        audioDir=params['inferenceDataPath']
        allFiles = os.listdir(audioDir)
        files = [name for name in allFiles if name.endswith('.wav') or name.endswith('.mp3')]
        
        for audioName in tqdm(files,desc='Loading audios'):
            
            waveform, sampleRate = torchaudio.load(audioDir+'/'+audioName)
            if sampleRate!=params['sampleRate']:
                trans=torchaudio.transforms.Resample(sampleRate,params['sampleRate'])
                waveform=trans(waveform)
                
            if waveform.size(0)!=1:
                waveform=waveform.mean(dim=0,keepdim=True)
            audios.append(waveform)
            
        melProcessor = torchaudio.transforms.MelSpectrogram(
                                sample_rate=sampleRate,
                                n_fft=fftSize,
                                win_length=windowSize,
                                hop_length=hopSize,
                                n_mels=melBands,
                                f_min=fmin,
                                f_max=fmax
                            ).to(params['inferenceDevice'])
        
        for i in trange(len(audios),desc='extracting mel spectrograms '):
            audios[i]=melProcessor(audios[i].to(params['inferenceDevice'])).clamp(min=2e-4).log2()
            
        g=Generator().to(params['inferenceDevice'])
        g.load_state_dict(torch.load(params['generatorPathInf']))
        
        g.eval()
        g.setDropKey(False)
        

        if params['schedulerPathInf']==None:
            for i in trange(len(audios),desc='Inferencing '):
                torchaudio.save(params['inferenceSavePath']+'/'+files[i],g(audios[i])[0].cpu(),sampleRate)
        else:
            refiner=Refiner().to(params['inferenceDevice'])
            refiner.load_state_dict(torch.load(params['refinerPathInf']))
            scheduler=Scheduler().to(params['inferenceDevice'])
            scheduler.load_state_dict(torch.load(params['schedulerPathInf']))
            
            for i in trange(len(audios),desc='Inferencing '):
                torchaudio.save(params['inferenceSavePath']+'/'+files[i],generate(g,refiner,audios[i],scheduler.getScheduler(),True,scheduler.getScaler())[0].cpu(),sampleRate)
       
            

if __name__=='__main__':
    inference()
    
