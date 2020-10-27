x = NRwaveforms;
p = 1;
q = 2;
waveforms = resample(x,p,q,'Dimension',2);

% save('LTEDataset_upSamp.mat','waveforms')
% save('NRDataset_downSamp.mat','waveforms')