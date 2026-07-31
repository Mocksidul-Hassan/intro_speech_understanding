import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)
    
    # ফ্রেম বানাই এবং এনার্জি হিসাব করি
    num_frames = 1 + (len(waveform) - frame_length) // step
    energy = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * step
        frame = waveform[start : start + frame_length]
        energy[i] = np.sum(frame ** 2)
    
    threshold = 0.1 * np.max(energy)
    
    # হাই-এনার্জি ফ্রেমগুলো খুঁজে সেগমেন্ট বানাই
    segments = []
    in_segment = False
    start_frame = 0
    
    for i in range(num_frames):
        if energy[i] > threshold:
            if not in_segment:
                in_segment = True
                start_frame = i
        else:
            if in_segment:
                in_segment = False
                start_sample = start_frame * step
                end_sample = i * step + frame_length
                segments.append(waveform[start_sample:end_sample])
    
    # যদি শেষ পর্যন্ত সেগমেন্ট চলতে থাকে
    if in_segment:
        start_sample = start_frame * step
        segments.append(waveform[start_sample:])
    
    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)
    
    for seg in segments:
        # Pre-emphasis
        pre = np.zeros_like(seg)
        pre[0] = seg[0]
        pre[1:] = seg[1:] - 0.97 * seg[:-1]
        
        # Frames + FFT
        num_frames = 1 + (len(pre) - frame_length) // step
        if num_frames < 1:
            num_frames = 1
        
        spectra = []
        for i in range(num_frames):
            start = i * step
            frame = pre[start : start + frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            
            X = np.abs(np.fft.fft(frame))
            # শুধু low-frequency half
            half = X[:frame_length // 2]
            spectra.append(half)
        
        # Average
        avg = np.mean(spectra, axis=0)
        # log spectrum
        model = np.log(np.maximum(avg, 1e-10))
        models.append(model)
    
    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)
    
    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K))
    
    for y in range(Y):
        for k in range(K):
            a = models[y]
            b = test_models[k]
            sims[y, k] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    test_outputs = []
    for k in range(K):
        best = np.argmax(sims[:, k])
        test_outputs.append(labels[best])
    
    return sims, test_outputs