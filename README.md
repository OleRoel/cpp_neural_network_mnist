# cpp_neural_network_mnist
This is a coding experiment to compare speed of a C++ implementation for training a MNIST network against a Python implementation using Numpy/Scikit in a Jupyter notebook.

The original code can be found in the "Code for the Make Your Own Neural Network book" in Tariq Rashid's repository here: https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork

The MNIST datasets for training and testing the neural network can be found here: https://pjreddie.com/projects/mnist-in-csv/

The same training is performed with different flavours of the same code.

### BLAS
Uses [BLAS](http://www.netlib.org/blas/) library for optimizing matrix operations with help of the cblas library. The library is preinstalled on my OS.

### BLIS
[BLIS](https://github.com/flame/blis) is a portable software framework for instantiating high-performance BLAS-like dense linear algebra libraries. The library has been cloned from github and then:

```bash
./configure --prefix=/usr/local --enable-cblas -t pthreads CFLAGS="-std=C11 -msse4.2 -mfpmath=sse -O3" CC=clang  auto
make -j8
sudo make install
```

### MKL
Uses [Intel® Math Kernel Library](https://software.intel.com/en-us/mkl) and their cblas library for matrix operations optimized against Intel processors.

### CUDA and cuBLAS
Uses [NVIDIA CUDA](https://en.wikipedia.org/wiki/CUDA) and [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) to perform training on a GPU.

## Building

**cblas:**
```sh
clang++ mnist.cpp -I /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/ -lcblas -std=c++17 -msse4.2 -mfpmath=sse -pthread -O3 -DTARGET_CBLAS
```

**blis:**
```sh
clang++ mnist.cpp -I /usr/local/include/blis /usr/local/lib/libblis.a -std=c++17 -msse4.2 -mfpmath=sse -pthread -O3 -DTARGET_CBLAS
```

**MKL:**
```sh
clang++ mnist.cpp ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl -std=c++17 -msse4.2 -mfpmath=sse -pthread  -DMKL_ILP64 -m64 -I${MKLROOT}/include -O3 -DTARGET_MKL 
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
|            | Performance | Train Time [s] | Test Time [s] |
| ---------- |------------:| ---------------:|-------------:|
| **cblas**  |      0.9668 |          37.192 |        0.200 |
| **blis**   |      0.9667 |          17.471 |        0.122 |
| **MKL**    |      0.9664 |          16.406 |        0.098 |
| **cuBLAS** |      0.9624 |          66.196 |        0.735 |
| **Python** |      0.9668 |         260.706 |        1.362 |

MKL is currently 4.125 times faster than cuBLAS (after refactoring and unifying the cblas/MKL code) – I guess there is room for improvement for my CUDA implementation.

Hardware used:<br>
MacBook Pro (15-inch, 2018), 2,9 GHz Intel Core i9<br>
GeForce GTX 1080 Ti

