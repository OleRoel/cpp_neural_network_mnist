/*
  C++ translation of python notebook using cuBLAS for Make Your Own Neural Network from Tariq Rashid
  https://github.com/makeyourownneuralnetwork
  code for a 3-layer neural network, and code for learning the MNIST dataset
  (c) Ole Roel, 2018
  license is GPLv3
*/

#include <numeric>
#include <chrono>

#if defined(TARGET_CBLAS) || defined(TARGET_MKL)
#include "includes/cblas/vector.hpp"
#elif defined(TARGET_CUBLAS)
#include "includes/cublas/vector.hpp"
#endif

#include "includes/layer.hpp"
#include "includes/images_buffer.hpp"
#include "includes/neural_network.hpp"

constexpr std::size_t inputnodes = 784;
constexpr std::size_t hiddennodes = 200;
constexpr int outputnodes = 10;
constexpr float learingrate = 0.01;

typedef NeuralNetwork<inputnodes, hiddennodes, outputnodes> MNIST_NEURAL_NETWORK;
typedef ImagesBuffer<inputnodes> IMAGES_BUFFER;

float t[10] = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
Vector<10> targets(t);

void run_training(MNIST_NEURAL_NETWORK& nn, const IMAGES_BUFFER& buff) {
    for (size_t i = 0; i < buff.size(); i++) {
        targets.set_at(buff.get_number_at(i), 0.99);
        nn.train(buff.get_image_array_at(i), targets);
        targets.set_at(buff.get_number_at(i), 0.01);
    }
}

void run_test(const MNIST_NEURAL_NETWORK& nn, const IMAGES_BUFFER& buff) {
    std::vector<int> scorecard;

    std::vector<float> outputs(outputnodes);

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
    std::cout << "performance = " << std::setw(6) << float(sum)/float(scorecard.size()) << std::endl;
}

int main(void)
{
    constexpr int epochs = 10;
    auto start_time = std::chrono::high_resolution_clock::now();

    MNIST_NEURAL_NETWORK nn(learingrate);
    {
        IMAGES_BUFFER train_buff("mnist_train.csv", 60e3);
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
        IMAGES_BUFFER test_buff("mnist_test.csv", 10e3);
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