# Guass Rank Scaler
  
A scikit-learn style transformer that scales numeric variables to normal distributions. 

Input normalization for neural networks is very important. GaussRank is an effective algorithm for converting numeric variable distributions to normals. It is based on rank transformation. The first step is to assign a spacing between -1 and 1 to the sorted features, then apply the inverse of error function `erfinv` to make it look like a Gaussian. This generally works much better than Standard or MinMax Scaler.
  
## Important Links
  
* [Interview of the Kaggle competition winner (Michael Jahrer)](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629#250927)  
* [Blog post introducing GuassRank's concept and simple implementation (Zygmunt Zając)](http://fastml.com/preparing-continuous-features-for-neural-networks-with-rankgauss)
  
## Usage
