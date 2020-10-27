x = NRwaveforms;
p = 1;
q = 2;
waveforms = resample(x,p,q,'Dimension',2);

% save('NRDataset2.mat','NRwaveforms')
% save('LTEDataset2.mat','LTEwaveforms')