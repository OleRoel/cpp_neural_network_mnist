/*
  C++ translation of python notebook for Make Your Own Neural Network from Tariq Rashid
  https://github.com/makeyourownneuralnetwork
  code for a 3-layer neural network, and code for learning the MNIST dataset
  (c) Ole Roel, 2018
  license is GPLv3
*/

#include <cblas.h>

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
constexpr std::size_t outputnodes = 10;
constexpr double learingrate = 0.01;

typedef NeuralNetwork<inputnodes, hiddennodes, outputnodes> MNIST_NEURAL_NETWORK;

// Some helpers for debugging
//
void print_vector(const std::string& name, const std::vector<double>& vec);
void print_matrix(const std::string& name, const std::vector<double>& matrix, int row, int column);
void read_matrix(const std::string& filename, std::vector<double>& result);

auto sigmoid = [](auto d) { return (1.0 / (1.0 + std::exp(-d))); };
auto derive = [](auto a, auto b) { return a * b * (1.0 - b); };

template<int M, int N> void feed_forward(const double* A, const double* x, std::vector<double>& result) {
    double* y {&result[0]};

    cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, 1.0, A, M, x, 1, 0.0, y, 1);
    std::transform(result.begin(), result.end(), result.begin(), sigmoid);
} 

template<int M, int N> void hidden_layer_error(const double* A, const double* x, std::vector<double>& result) {
    double* y {&result[0]};
    cblas_dgemv(CblasColMajor, CblasTrans, M, N, 1.0, A, M, x, 1, 0.0, y, 1);
}

template<int M, int N> void backpropagate(const std::vector<double>& vec_in,
                                            const std::vector<double>& vec_out,
                                            const std::vector<double>& vec_err,
                                            double learningrate,
                                            std::vector<double>& result) {
    std::vector<double> vec(M);
    std::transform(vec_err.begin(), vec_err.end(), vec_out.begin(), vec.begin(), derive);

    const double* v {&vec[0]};
    const double* w {&vec_in[0]};
    double* C {&result[0]};
    
    cblas_dger(CblasColMajor, M, N, learningrate, v, 1, w, 1, C, M);
}

void read_matrix(const std::string& filename, std::vector<double>& result) {
    std::ifstream infile(filename);
    std::string line;
    std::string delims = ",";
    std::vector<double>::iterator it = result.begin();

    while (std::getline(infile, line)) {
        std::vector<std::string> vec;
        boost::split(vec, line, boost::is_any_of(delims));

        it = std::transform(vec.begin(), vec.end(), it, [](const std::string& p) -> double { return std::stod(p); });
    }
}

void print_vector(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << ":\n";
    for (const double& i : vec) {
        std::cout << i <<  ", ";
    }
    std::cout << std::endl;
}

void print_matrix(const std::string& name, const std::vector<double>& matrix, int row, int column)
{
    std::cout << "Matrix " << name << " has " << row << " rows and " << column << " columns:\n";
    for (int i = 0; i < row; i++){
        for (int j = 0; j < column; j++) {
            std::cout << '[' << i << ',' << j << ']'<< std::setw(6) << matrix[i*column + j] << " ";
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

void dgemv(const double *A, const double *u, double *v, const int n, const int m) {
    for(int i=0; i<n; i++) {
        double sum = 0;
        for(int j=0; j<m; j++) {
            sum += A[m*i+j]*u[j];
        }
        v[i] = sum;
    }
}

template<int inputnodes, int hiddennodes, int outputnodes>
class NeuralNetwork
{
    private:
        double learningrate;

        std::vector<double> wih;
        std::vector<double> who;

        mutable std::vector<double> hidden_outputs;
        mutable std::vector<double> outputs;

        std::vector<double> hidden_errors;
        std::vector<double> output_errors;

    public:
        NeuralNetwork(double learningrate) :
            learningrate{learningrate},
            wih(inputnodes*hiddennodes),
            who(outputnodes*hiddennodes),
            hidden_outputs(hiddennodes),
            outputs(outputnodes),
            hidden_errors(hiddennodes),
            output_errors(outputnodes)
            {
#if 1
                read_matrix("wih_col_major.csv", wih);
                read_matrix("who_col_major.csv", who);
#else
                std::random_device rd{};
                std::mt19937 gen{rd()};

                {
                    double sigma = std::pow(inputnodes, -0.5);
                    std::normal_distribution<> d{0.0, sigma};
                    std::generate(wih.begin(), wih.end(), [&gen, &d]() mutable { return d(gen); });
                }
                {
                    double sigma = std::pow(outputnodes, -0.5);
                    std::normal_distribution<> d{0.0, sigma};
                    std::generate(who.begin(), who.end(), [&gen, &d]() mutable { return d(gen); });
                }
#endif
            }
        ~NeuralNetwork() {
        }
    
    void train(const std::vector<double>& inputs, const std::vector<double>& target)
    {
        feed_forward<hiddennodes, inputnodes>(&wih[0], &inputs[0], hidden_outputs);
        feed_forward<outputnodes, hiddennodes>(&who[0], &hidden_outputs[0], outputs);

        std::transform(target.begin(), target.end(), outputs.begin(), output_errors.begin(), std::minus<double>());

        hidden_layer_error<outputnodes, hiddennodes>(&who[0], &output_errors[0], hidden_errors);

        backpropagate<outputnodes, hiddennodes>(hidden_outputs, outputs, output_errors, learningrate, who);
        backpropagate<hiddennodes, inputnodes>(inputs, hidden_outputs, hidden_errors, learningrate, wih);
    }

    const std::vector<double>& query(const std::vector<double>& inputs) const {
        feed_forward<hiddennodes, inputnodes>(&wih[0], &inputs[0], hidden_outputs);
        feed_forward<outputnodes, hiddennodes>(&who[0], &hidden_outputs[0], outputs);

        return outputs;
    }
};

void run_training(MNIST_NEURAL_NETWORK& nn) {
    std::ifstream infile("mnist_train.csv");
    std::string line;
    std::string delims = ",";
    // std::size_t ii {0};
    while (std::getline(infile, line)) {
        // if (ii++ % 100 == 0) {
        //     std::cout << "+" << std::flush;
        // }
        std::vector<std::string> vec;
        boost::split(vec, line, boost::is_any_of(delims));

        std::vector<double> input(784);
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

        if (vec.size() == 785) {
            for (int i = 0; i < inputnodes; ++i) {
                inputs[i] = (double(std::stoi(vec[i+1])) / 255.0 * 0.99) + 0.001;
            }
            int correct_label = std::stoi(vec[0]);

            const std::vector<double>& outputs = nn.query(inputs);
            int label = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));

            if (label == correct_label) {
                // std::cout << "found correct label: " << correct_label << std::endl;
                scorecard.push_back(1);
            } else {
                // std::cout << ":-( label: " << label << " should be: " << correct_label << std::endl;
                scorecard.push_back(0);
            }
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