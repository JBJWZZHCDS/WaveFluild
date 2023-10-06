# WaveFluild
params.py is for the hyperparameters,
inference.py is for generating audios,
train.py is for the model's training process,
schedulerTrain.py is for training a shorter sampling schedule.
Don't change models.py and modules.py.
Pth files with LJ are the models pretrained on LJspeech(sample rate=22050),
Pth files with VCTK are the models pretrained on VCTK(sample rate=22050).
Python==3.8.16,torch==2.0.1,torchaudio==2.0.2.