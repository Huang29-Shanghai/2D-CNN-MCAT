import mne
from mne.datasets import sample

fwd_fname = str(sample.data_path()) + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fwd_fname)
raw_fname = str(sample.data_path()) + '/MEG/sample/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname,preload=True)