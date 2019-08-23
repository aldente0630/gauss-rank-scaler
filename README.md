# Guass Rank Scaler
  
A scikit-learn style transformer that scales numeric variables to normal distributions. 

Input normalization for neural networks is very important. GaussRank is an effective algorithm for converting numeric variable distributions to normals. It is based on rank transformation. The first step is to assign a spacing between -1 and 1 to the sorted features, then apply the inverse of error function `erfinv` to make it look like a Gaussian. This generally works much better than Standard or MinMax Scaler.
  
## Important Links
  
* [Interview of the Kaggle competition winner (Michael Jahrer)](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629#250927)  
* [Blog post introducing GuassRank's concept and simple implementation (Zygmunt Zając)](http://fastml.com/preparing-continuous-features-for-neural-networks-with-rankgauss)
  
## Usage

Gauss Rank Scaler is a fully compatible sklearn transformer that can be used in pipelines or existing scripts. Supported input formats include numpy arrays and pandas dataframes. All columns passed to the transformer are properly scaled.

## Example

```python
from sklearn_preprocessing.gauss_rank_scaler import *
import pandas as pd
from sklearn.datasets import load_boston
%matplotlib inline

# data preparation
bunch = load_boston()
y_train = bunch.target[:250]
y_test = bunch.target[250:]
X_train = pd.DataFrame(bunch.data[:250], columns=bunch.feature_names)
X_test = pd.DataFrame(bunch.data[250:], columns=bunch.feature_names)

# use Gauss to encode two categorical features
enc = BinaryEncoder(cols=['CHAS', 'RAD']).fit(X)

# transform the dataset
numeric_dataset = enc.transform(X)
```
