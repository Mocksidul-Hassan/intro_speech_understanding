import numpy as np
import torch
import torch.nn as nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    
    '''
    # ========== 1. Pre-emphasis ==========
    pre = np.zeros_like(waveform)
    pre[0] = waveform[0]
    pre[1:] = waveform[1:] - 0.97 * waveform[:-1]
    
    # ========== 2. Features (spectrogram 4ms / 2ms) ==========
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)
    
    num_frames = 1 + (len(pre) - frame_length) // step
    features = []
    
    for i in range(num_frames):
        start = i * step
        frame = pre[start : start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        X = np.abs(np.fft.fft(frame))
        half = X[:frame_length // 2]
        features.append(half)
    
    features = np.array(features)
    
    # ========== 3. VAD labels (25ms / 10ms) ==========
    vad_frame = int(0.025 * Fs)
    vad_step = int(0.01 * Fs)
    
    num_vad = 1 + (len(waveform) - vad_frame) // vad_step
    energy = np.zeros(num_vad)
    for i in range(num_vad):
        start = i * vad_step
        frame = waveform[start : start + vad_frame]
        energy[i] = np.sum(frame ** 2)
    
    threshold = 0.1 * np.max(energy)
    
    
    labels_vad = np.zeros(num_vad, dtype=int)
    current_label = 1
    in_segment = False
    
    for i in range(num_vad):
        if energy[i] > threshold:
            if not in_segment:
                in_segment = True
            labels_vad[i] = current_label
        else:
            if in_segment:
                in_segment = False
                current_label += 1
    
  
    # 10ms / 2ms = 5
    labels = np.zeros(num_frames, dtype=int)
    for i in range(num_vad):
        start_f = i * 5
        end_f = min(start_f + 5, num_frames)
        labels[start_f:end_f] = labels_vad[i]
    
    
    if num_frames > len(labels):
        labels = np.pad(labels, (0, num_frames - len(labels)), constant_values=0)
    labels = labels[:num_frames]
    
    return features, labels

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors
    labels (NFRAMES) - numpy array of labels (integers)
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    NFEATS = features.shape[1]
    n_classes = int(1 + np.max(labels))
    
    model = nn.Sequential(
        nn.LayerNorm(NFEATS),
        nn.Linear(NFEATS, n_classes)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    lossvalues = np.zeros(iterations)
    
    model.train()
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        lossvalues[i] = loss.item()
    
    return model, lossvalues

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    model.eval()
    X = torch.tensor(features, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
    
    return probs.detach().numpy()