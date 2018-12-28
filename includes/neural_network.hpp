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

        void feed_forward(const Vector<inputnodes>& inputs) const {
            hidden_layer.feed_forward(inputs);
            output_layer.feed_forward(hidden_layer.get_neurons());
        }

        void updated_errors(const Vector<outputnodes>& targets) {
            output_layer.calc_errors(targets);
            output_layer.backpropagate_errors(hidden_layer.get_errors());
        }

        void backpropagate(const Vector<inputnodes>& inputs) {
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
    
    void train(const Vector<inputnodes>& inputs, const Vector<outputnodes>& targets) {
        feed_forward(inputs);
        updated_errors(targets);
        backpropagate(inputs);
    }

    const std::vector<double>& query(const Vector<inputnodes>& inputs, std::vector<double>& _outputs) const {
        feed_forward(inputs);

        output_layer.get_neurons().get(_outputs);

        return _outputs;
    }
};


#endif