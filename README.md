# Guass Rank Scaler
  
A scikit-learn style transformer that scales numeric variables to a normal distribution. 

Input normalization for neural networks is critical while it does not matter with tree ensemble methods. GaussRank The best what I found during the past and works straight of the box is “RankGauss”. Its based on rank transformation.

First step is to assign a linspace to the sorted features from -1…1, then apply the inverse of error function ErfInv to shape them like gaussians, then I substract the mean. Binary features are not touched with this trafo (eg. 1-hot ones). This works usually much better than standard mean/std scaler or min/max.
  
## Important Links
  
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629#250927  
http://fastml.com/preparing-continuous-features-for-neural-networks-with-rankgauss  
  
## Usage


