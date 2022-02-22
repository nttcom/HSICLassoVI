# Effective Nonlinear Feature Selection Method based on HSIC Lasso and with Variational Inference

This repository implements "Effective Nonlinear Feature Selection Method based on HSIC Lasso and with Variational Inference" (HSICLassoVI) at AISTATS 2022.

## Requirements
- Python 3.6+
- See [requirements.txt](./requirements.txt) for the required Python packages.


## Installation

```sh
$ pip install -r requirements.txt
$ python setup.py install
```


## Usage

### Proposed (HSIC)

```python
class HSICLassoVI.models.api.Proposed_HSIC_Lasso(lam, tol=1e-5, nu=1.5, numiter=100, objhowoften=1)
```

#### Parameters

- `lam`: list
  - Regularization parameter. Input in the form `[negative domain, positive domain]`. Actually, due to the non-negative constraint, the regularization coefficient in the negative domain is infinite with the proposed method, so the input is `[np.inf, float]`
- `tol`: float (optional), default=1e-5
  - Tolerance for stopping criterion.
- `nu`: float (optional), default=1.5
  - Shape parameter of the prior distribution.
- `number`: int (optional), default=100
  - Maximum number of iterations
- `objhowoften`: int (optional), default=1
  - Frequency of calculating the marginal likelihood.

#### Attributes

- `KX`: ndarray of shape (n_samples * B * M, n_features, 1)
  - Design matrix
- `Ky`: ndarray of shape (n_samples * B * M, 1)
  - Response vector

- `omega`: ndarray of shape (n_features, 1)
  - Parameter vecto
- `eta`: ndarray of shape (n_features, )
  - Weights assigned to the features 
- `bound`: float
  - variational lower bound
- `sigma`: float
  - Standard deviation of likelihood

#### Methods

- `input(X, y, featname)`
  - X: ndarray of shape (n_samples, n_features)
    - Input data
  - y: ndarray of shape (n_samples, )
    - Output data
  - featname (optional)
    - Feature names

- `classification_multi(B=20, M=3, n_jobs=-1, kernels=['Gaussian'])`
  - B: int (optional), default=20
    - Block parameter of the block HSIC Lasso
  - M: int (optional), default=3
    - Permutation parameter of the block HSIC Lasso
    - Note: `B=0` and `M=1` is the vanilla HSIC Lasso
  - n_jobs: int (optional), default=-1
    - Number of parallel computations of the kernel matrices
  - kernels: list (optional), default=['Gaussian']
    - Kernel function of input data

- `regression_multi(B=20, M=3, n_jobs=-1, kernels=['Gaussian'])`
  - B: int (optional), default=20
    - Block parameter of the block HSIC Lasso
  - M: int (optional), default=3
    - Permutation parameter of the block HSIC Lasso
    - Note: `B=0` and `M=1` is the vanilla HSIC Lasso
  - n_jobs: int (optional), default=-1
    - Number of parallel computations of the kernel matrices
  - kernels: list (optional), default=['Gaussian']
    - Kernel function of input data

- `get_index_score()`
  - Regression coefficients sorted by relevance.

- `get_features()`
  - Feature names sorted by relevance.

### Proposed (NOCCO)

```python
class HSICLassoVI.models.api.Proposed_NOCCO_Lasso(lam, tol=1e-5, nu=1.5, numiter=100, objhowoften=1, eps=0.001)
```

This class has the same Parameters, Attributes and Methods as `Proposed_HSIC_Lasso`, except for the following.

#### Parameters

- `eps`: float, default=1e-3
  - Regularization parameter for NOCCO
  
  
## Examples

[Example.ipynb](./examples/Example.ipynb) contains a specific execution example (Additional experimental results: TBA).

The results from the Camera-Ready Paper were obtained in Ubuntu 18.04 server with 96-core Intel Xeon Platinum 2.7 GHz and 1.5 TB RAM memory.
Corresponding to the Camera-Ready Paper, synthetic datasets can be generated using [Example.ipynb](./examples/Example.ipynb), and real-world data can be downloaded from [Gas Sensor](https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset) and [USPS](https://jundongl.github.io/scikit-feature/datasets.html).

We provide the minimal code to reproduce some of the experiments performed. The reason is to make the code easier to track. If you want more information about some parts of the implementation feel free to email to [kazuki.koyama@ntt.com](kazuki.koyama@ntt.com).

  
## Notice

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Reference

We compare our proposed method with the following implementations.
- [pyHSICLasso](https://github.com/riken-aip/pyHSICLasso)
- [fsnet](https://github.com/singh-ml/fsnet)

