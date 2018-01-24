import util
import numpy as np

MOVEMENT_START = 1*160 # MI starts 1s after trial begin
MOVEMENT_END = 5*160 # MI lasts 4 seconds
NOISE_LEVEL = 0.01

def load_raw_data(electrodes, subject=None, num_classes=2, long_edge=False):
    # load from file
    trials = []
    labels = []
    
    if subject == None:
        subject_ids = range(1,110)
    else:
        try:
            subject_ids = [int(subject)]
        except:
            subject_ids = subject
    
    for subject_id in subject_ids:
        try:
            t, l, loc, fs = util.load_physionet_data(subject_id, num_classes, long_edge=long_edge)
            if num_classes == 2 and t.shape[0] != 42:
                # drop subjects with less trials
                continue
            trials.append(t[:,:, electrodes])
            labels.append(l)
        except:
            pass
            
    
    return np.array(trials).reshape((len(trials),) + trials[0].shape + (1,)), np.array(labels)

def split_idx(run, nruns, idx, seed=1337):
    """
    Shuffle and split a list of indexes into training and test data with a fixed 
    random seed for reproducibility
    
    run: index of the current split (zero based)
    nruns: number of splits (> run)
    idx: list of indices to split
    """
    rs = np.random.RandomState()
    rs.seed(seed)
    rs.shuffle(idx)
    n = len(idx)
    start = int(np.floor(float(run)/nruns*n))
    end = int(np.ceil(float(run+1)/nruns*n))
    train_idx = idx[:start] + idx[end:]
    test_idx = idx[start:end]
    return train_idx, test_idx

def crossval_test(X, y, test_idx, seg_length, flatten=True, fix_offset=None):
    """
    Prepares a test set of (X, y) with the subjects included in test_idx. 
    
    flatten: if True output shape is (N, seg_length, N_channels), otherwise
    output shape is (N_subjects, N_trials, seg_length, N_channels) for 
    per-subject validation. 
    fix_offset: Set the offset of the segment from the start of a trial to
    a selected value. Otherwise samples start at the cue.
    """
    ntrials = X.shape[1]
    
    if flatten:
        preshape = (len(test_idx) * ntrials,)
    else:
        preshape = (len(test_idx), ntrials)
    
    Xout = np.zeros(preshape + (seg_length,) + X.shape[-2:])
    yout = np.zeros(preshape + (y.shape[-1],))

    for i, subject in enumerate(test_idx):
        for j in range(ntrials):
            trial = X[subject, j, :, :]
            
            if fix_offset:
                offset = fix_offset
            else:
                # find a reasonable starting index depending on desired length
                rel_start = np.minimum(0, MOVEMENT_END-MOVEMENT_START-seg_length)
                offset = np.maximum(0, MOVEMENT_START + rel_start)
               
            # normalize based on per-channel mean 
            x = trial[offset:offset+seg_length , :]
            mu = x.mean(0).reshape((1,)+x.shape[1:])
            sigma = np.maximum(x.std(0).reshape((1,)+x.shape[1:]), 1e-10)

            out = (x - mu) / sigma

            if flatten:
                ix = i*ntrials+j
                Xout[ix, :, :] = out
                yout[ix, :] = y[subject, j, :]
            else:
                Xout[i, j, :, :] = out
                yout[i, j, :] = y[subject, j, :]
    return Xout,yout

def crossval_gen(X, y, train_idx, seg_length, batch_size):
    """
    Generator that produces training batches in an infinite loop by 
    randomly selecting them from the training data, normalizing them,
    and adding a little noise
    """
    while True:
        Xout = np.zeros((batch_size, seg_length) + X.shape[-2:])
        yout = np.zeros((batch_size,) + (y.shape[-1],))
            
        for i in range(batch_size):
            # randomly choose subject and trial
            subject = np.random.choice(train_idx)
            trial = np.random.randint(0, X.shape[1])
            
            # find a reasonable starting index depending on desired length
            rel_start = np.minimum(0, MOVEMENT_END-MOVEMENT_START-seg_length)
            offset = np.maximum(0, MOVEMENT_START + rel_start)

            # normalize based on per-channel mean 
            x = X[subject, trial, offset:offset+seg_length , :]
            mu = x.mean(0).reshape((1,)+x.shape[1:])
            sigma = np.maximum(x.std(0).reshape((1,)+x.shape[1:]), 1e-10)
            
            add_noise = NOISE_LEVEL * np.random.randn(seg_length, X.shape[-2], X.shape[-1])
            Xout[i, :, :] = (x - mu) / sigma + add_noise
            yout[i, :] = y[subject, trial, :]
        yield Xout, yout
