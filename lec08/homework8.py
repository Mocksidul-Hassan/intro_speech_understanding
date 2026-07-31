import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    frame_length (scalar) - length of the frame, in samples
    step (scalar) - step size, in samples
    
    @returns:
    frames (np.ndarray((num_frames, frame_length))) - waveform chopped into frames
       frames[m/step,n] = waveform[m+n] only for m = integer multiple of step
    '''
    num_frames = 1 + (len(waveform) - frame_length) // step
    frames = np.zeros((num_frames, frame_length))
    
    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start : start + frame_length]
    
    return frames

def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    
    @params:
    frames (np.ndarray((num_frames, frame_length))) - the speech samples
    
    @returns:
    mstft (np.ndarray((num_frames, frame_length))) - the magnitude short-time Fourier transform
    '''
    mstft = np.abs(np.fft.fft(frames, axis=1))
    return mstft

def mstft_to_spectrogram(mstft):
    '''
    Convert max(0.001*amax(mstft), mstft) to decibels.
    
    @params:
    stft (np.ndarray((num_frames, frame_length))) - magnitude short-time Fourier transform
    
    @returns:
    spectrogram (np.ndarray((num_frames, frame_length)) - spectrogram 
    
    The spectrogram should be expressed in decibels (20*log10(mstft)).
    np.amin(spectrogram) should be no smaller than np.amax(spectrogram)-60
    '''
    floor = 0.001 * np.amax(mstft)
    spectrogram = 20 * np.log10(np.maximum(floor, mstft))
    return spectrogram