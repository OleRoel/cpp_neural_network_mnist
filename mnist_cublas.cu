/*
  C++ translation of python notebook using cuBLAS for Make Your Own Neural Network from Tariq Rashid
  https://github.com/makeyourownneuralnetwork
  code for a 3-layer neural network, and code for learning the MNIST dataset
  (c) Ole Roel, 2018
  license is GPLv3
*/

#include <cublas_v2.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstddef>                                         /* size_t */
#include <fstream>
#include <boost/algorithm/string.hpp>                      /* split */
#include <boost/algorithm/string/classification.hpp>       /* is_any_of */
#include <chrono>

template<int inputnodes, int hiddennodes, int outputnodes> class NeuralNetwork;

constexpr std::size_t inputnodes = 784;
constexpr std::size_t hiddennodes = 200;
constexpr int outputnodes = 10;
constexpr double learingrate = 0.01;

typedef NeuralNetwork<inputnodes, hiddennodes, outputnodes> MNIST_NEURAL_NETWORK;

__device__ __forceinline__ double sigmoid(double a)
{
    return 1.0 / (1.0 + exp (-a));
}

__global__ void sigmoid_kernel(const double * __restrict__ src, 
                                double * __restrict__ dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = sigmoid (src[i]);
    }
}

__device__ __forceinline__ double derive(double a, double b)
{
    return a * b * (1.0 - b);
}

__global__ void derive_kernel(const double * __restrict__ a, const double * __restrict__ b, 
    double * __restrict__ dst, int len)
{
    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < len; i += stride) {
        dst[i] = derive(a[i], b[i]);
    }
}

template<int N> 
class CudaVector {
    private:
        double* v;
 
    public:
        CudaVector() {
            cudaMalloc(reinterpret_cast<void**>(&v), sizeof(double)*N);
        }
        CudaVector(const std::vector<double>& vec) : CudaVector() {
            set(vec);
        }
        ~CudaVector() {
            cudaFree(v);
        }

        operator const double* () const { return v; }
        operator double* () { return v; }

        const double* begin() const { return v; }
        double* begin()  { return v; }
        const double* end() const { return &v[N]; }
        double* end()  { return &v[N]; }

        CudaVector& minus(const CudaVector& vec, cublasHandle_t handle) {
            double minusOne = -1.0;
            cublasDaxpy(handle, N, &minusOne, vec, 1, v, 1);
            return *this;
        }

        CudaVector& plus(const CudaVector& vec, cublasHandle_t handle) {
            double one = 1.0;
            cublasDaxpy(handle, N, &one, vec, 1, v, 1);        
            return *this;
        }

        inline CudaVector& operator = (std::vector<double>& vec) {
            set(vec);
        }

        inline CudaVector& operator -= (const CudaVector& vec) {
            return minus(vec);
        }

        inline CudaVector& operator += (const CudaVector& vec) {
            return plus(vec);
        }

        inline void get(std::vector<double>& vec) const {
            cudaMemcpy(&vec[0], v, N*sizeof(double), cudaMemcpyDeviceToHost);
        }

        inline void set(const std::vector<double>& vec) {
            cudaMemcpy(v, &vec[0], N*sizeof(double), cudaMemcpyHostToDevice);
        }

        void sigmoid() {
            /* Compute execution configuration */
            dim3 dimBlock(256);
            int threadBlocks = (N + (dimBlock.x - 1)) / dimBlock.x;
            if (threadBlocks > 65520)
                threadBlocks = 65520;
            dim3 dimGrid(threadBlocks);

            sigmoid_kernel<<<dimGrid,dimBlock>>>(v, v, N);
        }

        inline void derive(const CudaVector<N>& v1, const CudaVector<N>& v2) {
            /* Compute execution configuration */
            dim3 dimBlock(256);
            int threadBlocks = (N + (dimBlock.x - 1)) / dimBlock.x;
            if (threadBlocks > 65520)
            threadBlocks = 65520;
            dim3 dimGrid(threadBlocks);

            derive_kernel<<<dimGrid, dimBlock>>>(v1, v2, v, N);
        }

        void random() {
            std::random_device rd{};
            std::mt19937 gen{rd()};

            double sigma {std::pow(N, -0.5)};
            std::normal_distribution<> d{0.0, sigma};
            std::vector<double> vec(N);
            std::generate(vec.begin(), vec.end(), [&gen, &d]() mutable { return d(gen); });
            set(vec);
        }

        void print(const std::string& name) {
            std::cout << name << ":\n" << *this << std::endl;
        }

        void print(const std::string& name, int row, int column) {
            std::vector<double> matrix(N);
            get(matrix);

            std::cout << "Matrix " << name << " has " << row << " rows and " << column << " columns:\n";
            for (int i = 0; i < row; i++){
                for (int j = 0; j < column; j++) {
                    std::cout << '[' << i << ',' << j << ']'<< std::setw(6) << matrix[i*column + j] << " ";
                }
                std::cout << '\n';
            }
            std::cout << std::endl;
        }

        void read(const std::string& filename) {
            std::ifstream infile(filename);
            infile >> *this;
        }

        inline int size() const {
            return N;
        }
};

template<int N> 
std::ostream& operator << (std::ostream& os, const CudaVector<N>& obj) {
    std::vector<double> vec(obj.size());
    obj.get(vec);

    for (const double& i : vec) {
        os << i <<  ", ";
    }

    return os;
}

template<int N> 
std::istream& operator >> (std::istream& is, CudaVector<N>& obj) {
    std::vector<double> vec(obj.size());
    std::string line;
    std::string delims = ",";
    std::vector<double>::iterator it = vec.begin();

    while (std::getline(is, line)) {
        std::vector<std::string> vec;
        boost::split(vec, line, boost::is_any_of(delims));

        it = std::transform(vec.begin(), vec.end(), it, [](const std::string& p) -> double { return std::stod(p); });
    }

    obj.set(vec);

    return is;
}

template<int M, int N> void feed_forward(cublasHandle_t handle,
                                            const CudaVector<M*N>& A,
                                            const CudaVector<N>& x,
                                            CudaVector<M>& y) {
    const double alpha = 1.0;
    const double beta = 0.0;

    cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha, A, M, x, 1, &beta, y, 1);
    y.sigmoid();
} 

template<int M, int N> void hidden_layer_error(cublasHandle_t handle, const double* A, const double* x, double* y) {
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemv(handle, CUBLAS_OP_T, M, N, &alpha, A, M, x, 1, &beta, y, 1);
}

template<int M, int N> void backpropagate(cublasHandle_t handle,
                                            const CudaVector<N>& vec_in,
                                            const CudaVector<M>& vec_out,
                                            const CudaVector<M>& vec_err,
                                            double learningrate,
                                            CudaVector<M*N>& result) {
    CudaVector<M> vec;
    vec.derive(vec_err, vec_out);
    cublasDger(handle, M, N, &learningrate, vec, 1, vec_in, 1, result, M);
}

class CuBLASBase {
    protected:
        cublasHandle_t handle;
    public:
        CuBLASBase() {
            cublasCreate(&handle);
        }
        ~CuBLASBase() {
            cublasDestroy(handle);
        }
};

template<int inputnodes, int hiddennodes, int outputnodes>
class NeuralNetwork : CuBLASBase
{
    private:
        double learningrate;

        CudaVector<inputnodes*hiddennodes> wih;
        CudaVector<outputnodes*hiddennodes> who;

        mutable CudaVector<hiddennodes> hidden_outputs;
        mutable CudaVector<outputnodes> outputs;

        mutable CudaVector<hiddennodes> hidden_errors;
        mutable CudaVector<outputnodes> output_errors;

        mutable CudaVector<inputnodes> inputs;

    public:
        NeuralNetwork(double learningrate) :
            CuBLASBase(),
            learningrate{learningrate}
            {
                cublasCreate(&handle);
#if 0
                wih.read("wih_col_major.csv");
                wih.read("who_col_major.csv");
#else
                wih.random();
                who.random();
#endif
            }
        ~NeuralNetwork() {
        }
    
    void train(const std::vector<double>& _inputs, const std::vector<double>& targets)
    {
        inputs.set(_inputs);
        output_errors.set(targets);

        feed_forward<hiddennodes, inputnodes>(handle, wih, inputs, hidden_outputs);
        feed_forward<outputnodes, hiddennodes>(handle, who, hidden_outputs, outputs);
        output_errors.minus(outputs, handle);
                
        hidden_layer_error<outputnodes, hiddennodes>(handle, who, output_errors, hidden_errors);

        backpropagate<outputnodes, hiddennodes>(handle, hidden_outputs, outputs, output_errors, learningrate, who);
        backpropagate<hiddennodes, inputnodes>(handle, inputs, hidden_outputs, hidden_errors, learningrate, wih);

#if 0
        hidden_outputs.print("hidden_outputs");
        outputs.print("outputs");
        output_errors.print("output_errors");
        hidden_errors.print("hidden_errors");
#endif
    }

    const std::vector<double>& query(const std::vector<double>& _inputs, std::vector<double>& _outputs) const {
        inputs.set(_inputs);

        feed_forward<hiddennodes, inputnodes>(handle, wih, inputs, hidden_outputs);
        feed_forward<outputnodes, hiddennodes>(handle, who, hidden_outputs, outputs);

        outputs.get(_outputs);

        return _outputs;
    }
};

void run_training(MNIST_NEURAL_NETWORK& nn) {
    std::ifstream infile("mnist_train.csv");
    std::string line;
    std::string delims = ",";
    // std::size_t ii {0};
    std::vector<std::string> vec;
    std::vector<double> input(784);

    while (std::getline(infile, line)) {
        // if (ii++ % 100 == 0) {
        //     std::cout << "+" << std::flush;
        // }
        boost::split(vec, line, boost::is_any_of(delims));

        for (std::size_t i = 0; i < inputnodes; ++i) {
            input[i] = (double(std::stoi(vec[i+1])) / 255.0 * 0.99) + 0.01;
        }
        std::vector<double> targets(outputnodes, 0.01);
        targets[std::stoi(vec[0])] = 0.99;

        nn.train(input, targets);
    }
    std::cout << std::endl;
}

void run_test(const MNIST_NEURAL_NETWORK& nn) {
    std::vector<int> scorecard;

    std::ifstream infile("mnist_test.csv");
    std::string line;
    std::string delims = ",";
    std::vector<std::string> vec;
    std::vector<double> inputs(784);

    while (std::getline(infile, line)) {
        boost::split(vec, line, boost::is_any_of(delims));

        for (int i = 0; i < inputnodes; ++i) {
            inputs[i] = (double(std::stoi(vec[i+1])) / 255.0 * 0.99) + 0.001;
        }
        int correct_label = std::stoi(vec[0]);

        std::vector<double> outputs(outputnodes);
        nn.query(inputs, outputs);
        int label = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));

        if (label == correct_label) {
            // std::cout << "found correct label: " << correct_label << std::endl;
            scorecard.push_back(1);
        } else {
            // std::cout << ":-( label: " << label << " should be: " << correct_label << std::endl;
            scorecard.push_back(0);
        }
    }
    
    int sum = std::accumulate(scorecard.begin(), scorecard.end(), 0);
    std::cout << "performance = " << std::setw(6) << double(sum)/double(scorecard.size()) << std::endl;
}

int main(void)
{
    constexpr int epochs = 10;

    MNIST_NEURAL_NETWORK nn(learingrate);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epochs; ++i)
    {
        std::cout << "Train epoch " << i << std::endl;

        run_training(nn);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    run_test(nn);
    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "training took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds" << std::endl;

    std::cout << "test took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count()
              << " milliseconds" << std::endl;
}
