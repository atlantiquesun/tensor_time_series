# Dow Jones 30 Dataset

## Source
[Yahoo! Finance Python API](https://pypi.org/project/yfinance/)

## Data
1. **dj30.mat**: variable 'X' is a matrix of the size (827, 30). It contains the close prices of the 30 stocks that constitute Dow Jones 30 Index from 2019-03-20 to 2022-06-29.
2. **dj30_transformed.mat**: we apply an orthogonal transformation to 'X' along the second dimension (size 30) and obtain a matrix of the same size as 'X'. 
  a. 'C': discrete cosine transform
  b. 'F': discrete fourier transform
  c. 'W': discrete wavelet transform
3. **dj30_target.mat**: variable 'X' is a matrix of the size (1, 828). It contains the close values of Dow Jones 30 Index from 2019-03-20 to 2022-06-30.
