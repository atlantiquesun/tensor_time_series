# Tensor Time Series Analysis

1. temporal_entropy_github.py: calculate the temporal marginal and conditional entropies of a tensor time series.
2. nasdaq100 (directory): 
  - nasdaq100.mat: original Nasdaq-100 dataset; contains the OHLC stock prices of 50 randomly selected stocks from the Nasdaq-100 index
  - nasdaq100_dct.mat: the Nasdaq-100 dataset transformed along the third dimension (OHLC dimension) using **discrete cosine transform**
  - nasdaq100_dft.mat: the Nasdaq-100 dataset transformed along the third dimension (OHLC dimension) using **discrete Fourier transform**
  - nasdaq100_dwt.mat: the Nasdaq-100 dataset transformed along the third dimension (OHLC dimension) using **discrete wavelet transform**
