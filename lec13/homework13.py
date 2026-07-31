import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    nframes = int((len(speech) - frame_length) / frame_skip)
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start : start + frame_length]
        
        # LPC coefficients (a[0] = 1)
        a = librosa.lpc(frame, order=order)
        A[i, :] = a
        
        # Residual (excitation)
        # residual = filter(a, [1], frame)
        residual = np.convolve(frame, a, mode='full')[:frame_length]
        excitation[i, :] = residual
    
    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    nframes = A.shape[0]
    duration = nframes * frame_skip
    synthesis = np.zeros(duration)
    
    # All-pole filter: y[n] = e[n] - a1*y[n-1] - a2*y[n-2] - ...
    order = A.shape[1] - 1
    y_hist = np.zeros(order)
    
    for i in range(nframes):
        a = A[i, :]
        for n in range(frame_skip):
            idx = i * frame_skip + n
            if idx >= len(e):
                break
            # y[n] = e[n] - sum(a[1:] * y_hist)
            val = e[idx] - np.dot(a[1:], y_hist)
            synthesis[idx] = val
            # update history
            y_hist = np.roll(y_hist, 1)
            y_hist[0] = val
    
    return synthesis

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    nframes = excitation.shape[0]
    frame_length = excitation.shape[1]
    
    # Gain = RMS of the last frame_skip samples of each excitation frame
    gain = np.zeros(nframes)
    for i in range(nframes):
        residual = excitation[i, frame_length - frame_skip : frame_length]
        gain[i] = np.sqrt(np.mean(residual ** 2))
    
    # Robot excitation: impulses every T0 samples, scaled by gain
    e_robot = np.zeros(nframes * frame_skip)
    for i in range(nframes):
        start = i * frame_skip
        # place impulses in this frame region
        for n in range(0, frame_skip, T0):
            if start + n < len(e_robot):
                e_robot[start + n] = -gain[i]
    
    return gain, e_robot