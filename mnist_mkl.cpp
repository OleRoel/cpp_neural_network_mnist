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
#include <list> 
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
class NormalDistribution {
    private:
        std::random_device rd;
        std::mt19937 gen{rd()};

        double sigma;
        std::normal_distribution<> d;
    public:
        NormalDistribution() :
            rd{},
            gen{rd()},
            sigma{std::pow(N, -0.5)},
            d{0.0, sigma} {

            }
        NormalDistribution(const NormalDistribution& nd) :
            rd{},
            gen{rd()},
            sigma{nd.sigma},
            d{0.0, sigma} {

            }
        ~NormalDistribution() {

        }

    double operator() () {
        return d(gen);
    }
};

template<int N> 
class MKLVector {
    private:
        double* v;
 
    public:
        MKLVector() : 
            v{static_cast<double*>(mkl_malloc(N*sizeof(double), 64))} {
        }
        MKLVector(const std::vector<double>& vec) :
            MKLVector() {
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
            // cblas_dcopy(N, v, 0, &vec[0], 0);
        }

        inline MKLVector& set(const std::vector<double>& vec) {
            std::memcpy(v, &vec[0], N*sizeof(double));
            // cblas_dcopy(N, &vec[0], 0, v, 0);

            return *this;
        }

        inline void sigmoid() {
            std::transform(begin(), end(), begin(), _sigmoid);
        }

        inline void derive(const MKLVector<N>& v1, const MKLVector<N>& v2) {
            std::transform(v1.begin(), v1.end(), v2.begin(), v, _derive);
        }

        inline void derive(const MKLVector<N>& vec) {
            std::transform(begin(), end(), vec.begin(), begin(), _derive);
        }

        void random() {
            NormalDistribution<N> nd;

            std::generate(begin(), end(), nd);
        }

        void print(const std::string& name) const {
            std::cout << name << ":\n" << *this << std::endl;
        }

        void print(const std::string& name, int row, int column) const {
            std::vector<double> matrix(N);
            get(matrix);

            std::cout << "Matrix " << name << " has " << row << " rows and " << column << " columns:\n";
            for (int i = 0; i < row; i++){
                for (int j = 0; j < column; j++) {
                    // std::cout << '[' << i << ',' << j << ']'<< std::setw(6) << matrix[i*column + j] << " ";
                    std::cout << std::setw(6) << matrix[i*column + j] << " ";
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
    for (std::size_t i = 0; i < N; i++) {
        os << obj[i] <<  ", ";
    }

    return os;
}

template<int N> 
std::istream& operator >> (std::istream& is, MKLVector<N>& obj)
{
    std::string line;
    std::string delims = ",";
    double* it = obj.begin();
    std::vector<std::string> vec(N);

    while (std::getline(is, line)) {
        boost::split(vec, line, boost::is_any_of(delims));
        it = std::transform(vec.begin(), vec.end(), it, [](const std::string& p) -> double { return std::stod(p); });
    }

    return is;
}

template<int M, int N>
class Layer {
    private:
        MKLVector<M*N> weights;
        MKLVector<M> neurons;

        MKLVector<M> errors;

        double learningrate;
        
    public:
        Layer(double learningrate) : learningrate{learningrate} {
            weights.random();
        }
        Layer(const std::string& filename, double learningrate) : learningrate{learningrate} {
            weights.read(filename);
        }

        void feed_forward(const MKLVector<N>& inputs) {
            cblas_dgemv(CblasColMajor, CblasNoTrans, M, N, 1.0, weights, M, inputs, 1, 0.0, neurons, 1);
            neurons.sigmoid();
        }

        void backpropagate_errors(MKLVector<N>& out_errors) {
            cblas_dgemv(CblasColMajor, CblasTrans, M, N, 1.0, weights, M, errors, 1, 0.0, out_errors, 1);
        }

        void backpropagate(const MKLVector<N>& inputs) {
            errors.derive(neurons);
            cblas_dger(CblasColMajor, M, N, learningrate, errors, 1, inputs, 1, weights, M);
        }

        inline const  MKLVector<M*N>& get_weights() const {
            return weights;
        }

        inline const MKLVector<M>& get_neurons() const {
            return neurons;
        }

        inline const MKLVector<M>& get_errors() const {
            return errors;
        }

        inline MKLVector<M>& get_errors() {
            return errors;
        }

        inline void calc_errors(const std::vector<double>& vec) {
            errors = vec;
            errors -= neurons;
        }
};

template<int inputnodes, int hiddennodes, int outputnodes>
class NeuralNetwork {
    private:
        mutable Layer<hiddennodes, inputnodes> hidden_layer;
        mutable Layer<outputnodes, hiddennodes> output_layer;

        mutable MKLVector<inputnodes> inputs;

        void feed_forward() const {
            hidden_layer.feed_forward(inputs);
            output_layer.feed_forward(hidden_layer.get_neurons());
        }

        void updated_errors(const std::vector<double>& expected_vals) {
            output_layer.calc_errors(expected_vals);
            output_layer.backpropagate_errors(hidden_layer.get_errors());
        }

        void backpropagate() {
            output_layer.backpropagate(hidden_layer.get_neurons());
            hidden_layer.backpropagate(inputs);
        }

    public:
        NeuralNetwork(double learningrate) :
            hidden_layer(learningrate),
            output_layer(learningrate) {
        }
        ~NeuralNetwork() {
        }
    
    void train(const std::vector<double>& _inputs, const std::vector<double>& targets) {
        inputs = _inputs;
    
        feed_forward();
        updated_errors(targets);
        backpropagate();
    }

    const std::vector<double>& query(const std::vector<double>& _inputs, std::vector<double>& _outputs) const {
        inputs = _inputs;

        feed_forward();

        output_layer.get_neurons().get(_outputs);

        return _outputs;
    }
};

class ImagesBuffer {
    private:
        typedef std::vector<double> image_array;
        std::vector<image_array> buffers;
        std::vector<int> numbers;

        void read(const std::string& filename) {
            std::ifstream infile(filename);
            std::string line;
            std::string delims = ",";
            std::vector<std::string> vec;
            image_array inputs(inputnodes);
            while (std::getline(infile, line)) {
                boost::split(vec, line, boost::is_any_of(delims));

                std::transform(vec.begin()+1, vec.end(), inputs.begin(), [](const std::string& p) -> double { return (double(std::stoi(p)) / 255.0 * 0.99) + 0.001; });

                int correct_label = std::stoi(vec[0]);

                buffers.push_back(inputs);
                numbers.push_back(correct_label);
            }
        }

    public:
        ImagesBuffer(const std::string& filename) {
            read(filename);
        }

        const std::vector<double>& get_image_array_at(std::size_t idx) const {
            return buffers[idx];
        }
        const int get_number_at(std::size_t idx) const {
            return numbers[idx];
        }

        std::size_t size() const {
            return numbers.size();
        }
};

void run_training(MNIST_NEURAL_NETWORK& nn, const ImagesBuffer& buff) {
    for (size_t i = 0; i < buff.size(); i++) {
        std::vector<double> targets(outputnodes, 0.01);
        targets[buff.get_number_at(i)] = 0.99;

        nn.train(buff.get_image_array_at(i), targets);
    }
}

void run_test(const MNIST_NEURAL_NETWORK& nn, const ImagesBuffer& buff) {
    std::vector<int> scorecard;

    std::vector<double> outputs(outputnodes);

    for (size_t i = 0; i < buff.size(); i++) {
        nn.query(buff.get_image_array_at(i), outputs);

        int label = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));

        if (label == buff.get_number_at(i)) {
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
    auto start_time = std::chrono::high_resolution_clock::now();

    MNIST_NEURAL_NETWORK nn(learingrate);
    {
        ImagesBuffer train_buff("mnist_train.csv");
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < epochs; ++i) {
            std::cout << "Train epoch " << i << std::endl; 

            run_training(nn, train_buff);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "training took "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " milliseconds" << std::endl;
    }

    {
        ImagesBuffer test_buff("mnist_test.csv");
        auto t1 = std::chrono::high_resolution_clock::now();
        run_test(nn, test_buff);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "test took "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                << " milliseconds" << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "total run-time "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count()
            << " milliseconds" << std::endl;
}