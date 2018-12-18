/*
  C++ translation of python notebook using cuBLAS for Make Your Own Neural Network from Tariq Rashid
  https://github.com/makeyourownneuralnetwork
  code for a 3-layer neural network, and code for learning the MNIST dataset
  (c) Ole Roel, 2018
  license is GPLv3
*/

#include <mkl.h>

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

auto _sigmoid = [](auto d) { return (1.0 / (1.0 + std::exp(-d))); };
auto _derive = [](auto a, auto b) { return a * b * (1.0 - b); };

template<int N> 
class MKLVector {
    private:
        double* v;
 
    public:
        MKLVector() {
            v = (double *)mkl_malloc( N*sizeof(double), 64 );
        }
        MKLVector(const std::vector<double>& vec) : MKLVector() {
            set(vec);
        }
        ~MKLVector() {
            mkl_free(v);
        }

        operator const double* () const { return v; }
        operator double* () { return v; }

        const double* begin() const { return v; }
        double* begin()  { return v; }
        const double* end() const { return &v[N]; }
        double* end()  { return &v[N]; }

        MKLVector& minus(const MKLVector& vec) {
            cblas_daxpy(N, -1.0, vec, 1, v, 1);
            return *this;
        }

        MKLVector& plus(const MKLVector& vec) {
            cblas_daxpy(N, 1.0, vec, 1, v, 1);        
            return *this;
        }

        inline MKLVector& operator = (const std::vector<double>& vec) {
            return set(vec);
        }

        inline MKLVector& operator -= (const MKLVector& vec) {
            return minus(vec);
        }

        inline MKLVector& operator += (const MKLVector& vec) {
            return plus(vec);
        }

        inline void get(std::vector<double>& vec) const {
            std::memcpy(&vec[0], v, N*sizeof(double));
        }

        inline MKLVector& set(const std::vector<double>& vec) {
            std::memcpy(v, &vec[0], N*sizeof(double));
            return *this;
        }

        inline void sigmoid() {
            std::transform(v, &v[N], v, _sigmoid);
        }

        inline void derive(const MKLVector<N>& v1, const MKLVector<N>& v2) {
            std::transform(v1.begin(), v1.end(), v2.begin(), v, _derive);
        }

        void random() {
            std::random_device rd{};
            std::mt19937 gen{rd()};

            double sigma {std::pow(N, -0.5)};
            std::normal_distribution<> d{0.0, sigma};
            std::generate(v, &v[N], [&gen, &d]() mutable { return d(gen); });
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
std::ostream& operator << (std::ostream& os, const MKLVector<N>& obj)
{
    std::vector<double> vec(obj.size());
    obj.get(vec);

    for (const double& i : vec) {
        os << i <<  ", ";
    }

    return os;
}

template<int N> 
std::istream& operator >> (std::istream& is, MKLVector<N>& obj)
{
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

template<int M, int N> void feed_forward(const MKLVector<M*N>& A,
                                            const MKLVector<N>& x,
                                            MKLVector<M>& y) {
    cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, 1.0, A, M, x, 1, 0.0, y, 1);
    y.sigmoid();
} 

template<int M, int N> void hidden_layer_error(const double* A, const double* x, double* y) {
    cblas_dgemv(CblasColMajor, CblasTrans, M, N, 1.0, A, M, x, 1, 0.0, y, 1);
}

template<int M, int N> void backpropagate(const MKLVector<N>& vec_in,
                                            const MKLVector<M>& vec_out,
                                            const MKLVector<M>& vec_err,
                                            double learningrate,
                                            MKLVector<M*N>& result) {
    MKLVector<M> vec;
    vec.derive(vec_err, vec_out);
    cblas_dger(CblasColMajor, M, N, learningrate, vec, 1, vec_in, 1, result, M);
}

template<int inputnodes, int hiddennodes, int outputnodes>
class NeuralNetwork {
    private:
        double learningrate;

        MKLVector<inputnodes*hiddennodes> wih;
        MKLVector<outputnodes*hiddennodes> who;

        mutable MKLVector<hiddennodes> hidden_outputs;
        mutable MKLVector<outputnodes> outputs;

        mutable MKLVector<hiddennodes> hidden_errors;
        mutable MKLVector<outputnodes> output_errors;

        mutable MKLVector<inputnodes> inputs;

    public:
        NeuralNetwork(double learningrate) :
            learningrate{learningrate} {
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
    
    void train(const std::vector<double>& _inputs, const std::vector<double>& targets) {
        inputs = _inputs;
        output_errors = targets;

        feed_forward<hiddennodes, inputnodes>(wih, inputs, hidden_outputs);
        feed_forward<outputnodes, hiddennodes>(who, hidden_outputs, outputs);

        output_errors -= outputs;
                
        hidden_layer_error<outputnodes, hiddennodes>(who, output_errors, hidden_errors);

        backpropagate<outputnodes, hiddennodes>(hidden_outputs, outputs, output_errors, learningrate, who);
        backpropagate<hiddennodes, inputnodes>(inputs, hidden_outputs, hidden_errors, learningrate, wih);
    }

    const std::vector<double>& query(const std::vector<double>& _inputs, std::vector<double>& _outputs) const {
        inputs.set(_inputs);

        feed_forward<hiddennodes, inputnodes>(wih, inputs, hidden_outputs);
        feed_forward<outputnodes, hiddennodes>(who, hidden_outputs, outputs);

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