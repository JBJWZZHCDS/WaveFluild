params=dict(
    #params for mels
    melBands=80,
    fftSize=10,# means 2^10, FFT usually uses 2-power window
    windowSize=10, # means 2^10
    hopSize=8, # means 2^8 
    melTrainWindow=32,
    sampleRate=22050,
    fmin=0,
    fmax=8000,
    
    #params for training
    learnRateG=2e-4,
    learnRateR=2e-4,
    learnRateD=2e-4,
    gamma=0.99,
    betas=(0.8,0.99),
    weightDecay=0.01,
    trainDataPath='C:/deep_learning/speech_synthesis/LJSPeech/LJSpeech_1.1/wavs',
    generatorPath='./generator.pth',
    refinerPath='./refiner.pth',
    discriminatorPath='./discriminator.pth',
    trainEpoch=100,
    dropKeyEndEpoch=20,
    trainBatch=8,
    trainDevice='cuda:0',
    melScale=50,
    dwtScale=8,
    featureScale=15,
    timeDecayUp=1,
    timeDecayDown=0.5,
    timeStep=10,
    
    #params for scheduler
    schedulerEpoch=10,
    schedulerStep=4,
    schedulerLearnRate=0.1,
    schedulerTrainBatch=8,
    generatorPathSch='./generator.pth',
    refinerPathSch='./refiner.pth',
    discriminatorPathSch='./discriminator.pth',
    schedulerPath='./scheduler.pth',
    schedulerTrainDataPath='C:/deep_learning/speech_synthesis/LJSPeech/LJSpeech_1.1/wavs',
    schedulerTrainDevice='cuda:0',
    
    #params for inference
    inferenceDataPath='C:/deep_learning/speech_synthesis/unseen',
    generatorPathInf='./generatorVCTK.pth',
    refinerPathInf='./refinerVCTK.pth',
    inferenceDevice='cuda:0',
    schedulerPathInf='./schedulerVCTK.pth',
    inferenceSavePath='./inference',
    
    #params for model
    channelsGen=[256,128,64,64],
    
    attGroupsGen=[1,1,1,1],
    patchesGen=[[2,3],[2,3],[2,3,5],[2,3,5,7]],
    attMidChannelsGen=[64,32,16,16],
    dropRateGen=[0.3,0.25,0.2,0.15],
    
    resGroupsGen=[1,1,1,1],
    resMidChannelsGen=[256,128,64,64],
   
    upSampleRates=[8,8,4],
    upGroups=[1,1,1],
    
    downSampleRatesRefiner=[8,4,4,2],
    channelsRefiner=[32,64,128,256,256],
    groupsRefiner=[4,8,16,32],
    
    channelsDis=[64,128,256,512,1024],
    
    attGroupsDis=[4,8,16,32,64],
    patchesDis=[[2,3,5,7,11],[2,3,5,7],[2,3,5,7],[2,3,5],[2,3,5]],
    attMidChannelsDis=[32,64,128,256,512],
    dropRateDis=[0.3,0.25,0.2,0.15,0.1],
    
    resGroupsDis=[4,8,16,32,64],
    resMidChannelsDis=[64,128,256,512,1024],
   
    downSampleRates=[2,2,4,4],
    downGroups=[8,16,32,64],
)