import torch
import torch.nn.functional as func
import torchaudio
from tqdm import tqdm
from params import params
import os
import math


class AudioMelSet(torch.utils.data.DataLoader):
    def __init__(self,audioDir,toDevice,maxScale=None):
        
        with torch.no_grad():
            self.toDevice=toDevice
            self.sampleRate = 0
            self.melBands = params['melBands']
            self.fftSize = 2** params['fftSize']
            self.windowSize = 2** params['windowSize']
            self.hopSize = 2** params['hopSize']
            self.melTrainWindow = params['melTrainWindow']
            self.fmin=params['fmin']
            self.fmax=params['fmax']
            self.audios = []
            self.audioTrainWindow=self.hopSize*self.melTrainWindow
            self.maxLen = 0
            self.maxScale = maxScale
            self.shift = 32
            
            allFiles = os.listdir(audioDir)
            files = [name for name in allFiles if name.endswith('.wav') or name.endswith('.mp3')]
            for audioName in tqdm(files,desc='Loading audios'):
                waveform, self.sampleRate = torchaudio.load(audioDir+'/'+audioName)
                self.audios.append(waveform)
                self.maxLen=max(self.maxLen,waveform.size(-1))
                
            self.melProcessor = torchaudio.transforms.MelSpectrogram(
                                    sample_rate=self.sampleRate,
                                    n_fft=self.fftSize,
                                    win_length=self.windowSize,
                                    hop_length=self.hopSize,
                                    n_mels=self.melBands,
                                    f_min=self.fmin,
                                    f_max=self.fmax
                                ).to(toDevice)
            
    def audio2logaudio(self,audio):
        scale=((audio.abs()+self.shift).log2()-math.log2(self.shift))/math.log2(self.shift+self.maxScale)
        return audio.sign()*scale


    def logaudio2audio(self,logaudio):
        scale=2**(logaudio.abs()*math.log2(self.maxScale+self.shift)+math.log2(self.shift))-self.shift
        return logaudio.sign()*scale
    
    def getMel(self,audios):
        eps=2e-4
        mel=self.melProcessor(audios[:,:,:-1])
        mel=torch.clamp(mel,min=eps).log2()
        #mel=(mel+1).log2()
        return mel.squeeze(1) 
    
    def __getitem__(self,index):
        with torch.no_grad():
            audio=self.audios[index].to(self.toDevice)
            eps=2e-4
            if audio.size(-1)<self.audioTrainWindow:
                audio=func.pad(audio,pad=[0,self.audioTrainWindow-audio.size(-1)]) 
                mel=self.melProcessor(audio[:,:-1])#keep mel's size correct     
                #print('short',mel.min(),mel.max(),mel.mean())
                mel=torch.clamp(mel,min=eps).log2()
                #mel=(mel+1).log2()
            else:
                pos=torch.randint(low=0,high=audio.size(-1)-self.audioTrainWindow+1,size=(1,)).item()
                audio=audio[:,pos:pos+self.audioTrainWindow]
                mel=self.melProcessor(audio[:,:-1])#keep mel's size correct
                #print('long',mel.min(),mel.max(),mel.mean())
                mel=torch.clamp(mel,min=eps).log2()
                #mel=(mel+1).log2()
            return audio,mel.squeeze(0)
             
    def __len__(self):
        return len(self.audios)