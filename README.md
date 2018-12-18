# cpp_neural_network_mnist
This is a coding experiment to compare speed of a C++ implementation for training a MNIST network against a Numpy/Scikit implementation in a Jupyter notebook.

The original code can be found in the "Code for the Make Your Own Neural Network book" in Tariq Rashid's repository here: https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork

The MNIST datasets for training and testing the neural network can be found here: https://pjreddie.com/projects/mnist-in-csv/

The same training is performed with different flavours of the same code.

### BLAS
Uses [BLAS](http://www.netlib.org/blas/) library for optimizing matrix operations with help of the cblas library.

### MKL
Uses [Intel® Math Kernel Library](https://software.intel.com/en-us/mkl) and their cblas library for matrix operations optimized against Intel processors.

### CUDA and cuBLAS
Uses [NVIDIA CUDA](https://en.wikipedia.org/wiki/CUDA) and [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) to perform training on a GPU.

## Building

**cblas:**
```sh
g++ mnist_cblas.cpp -I /usr/local/opt/openblas/include -lcblas -std=c++17 -msse4.2 -mfpmath=sse -pthread -O3
```

**MKL:**
```sh
g++ mnist_mkl.cpp ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl -std=c++17 -msse4.2 -mfpmath=sse -pthread -O3  -DMKL_ILP64 -m64 -I${MKLROOT}/include 
```

**CUDA:**
```sh
nvcc mnist_cublas.cu -lcublas -O3 -Xptxas -O3,-v
```

## Running

```sh
./a.out
```

## Performance
| Flavour| Performance | Train Time [s] | Test Time [s] |
| ------ |------------:| ---------------:|-------------:|
| cblas  |      0.9673 |          80.258 |        0.791 |
| MKL    |      0.9671 |          53.739 |        0.600 |
| cuBLAS |      0.9624 |          66.196 |        0.735 |
| Python |      0.9668 |         260.706 |        1.362 |

MKL needs 18% less time for the training than cuBLAS do – I guess there is room for improvement for my CUDA implementation.

Hardware used:<br>
MacBook Pro (15-inch, 2018), 2,9 GHz Intel Core i9<br>
GeForce GTX 1080 Ti

