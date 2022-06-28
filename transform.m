load mlds_data/nasdaq100.mat;
N = numel(X);
I = size(X{1});
XX = {};

for i=1:N
    XX{i}=X{i}'; 
end

%% DCT transform
for k=1:N
    for p=1:I(1)
        A{k}(p,:)=dct(XX{k}(p,:));
    end    
end

%% DWT transform
B = Mydwt(XX);

%% DFT transform
for k=1:N
    for p=1:I(1)
        C{k}(p,:)=fft(XX{k}(p,:));
    end
end

save('mlds_data/nasdaq100_dct.mat', 'A');
save('mlds_data/nasdaq100_dwt.mat', 'B');
save('mlds_data/nasdaq100_dft.mat', 'C');
