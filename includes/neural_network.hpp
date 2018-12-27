#ifndef _NEURAL_NETWORK
#define _NEURAL_NETWORK

// #include "mkl/vector.hpp"
#include "layer.hpp"
#include <vector>

template<int> class Vector;

template<int inputnodes, int hiddennodes, int outputnodes>
class NeuralNetwork {
    private:
        mutable Layer<hiddennodes, inputnodes> hidden_layer;
        mutable Layer<outputnodes, hiddennodes> output_layer;

        mutable Vector<inputnodes> inputs;

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


#endif