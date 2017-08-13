#include "Matrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkOptimizer.hpp"
#include "NeuralNetworkSaver.hpp"
// Define layers using a custom matrix class.
typedef SigmoidFCL<Matrix_F> SigmoidFCL_F;
typedef LeakyReLUFCL<Matrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = ai::NeuralNetwork<Matrix_F, Layers...>;
// Define an optimizer using a custom matrix class.
typedef ai::NeuralNetworkOptimizer<Matrix_F, ai::mse_prime<Matrix_F>> NeuralNetworkOptimizer_F;

int main() {
    // Create some layers.
    SigmoidFCL_F inputLayer(784, 30);
    SigmoidFCL_F outputLayer(30, 10);
    // Create the network.
    NeuralNetwork_F<SigmoidFCL_F, SigmoidFCL_F> digitRecognizer(inputLayer, outputLayer);
    // Create an optimizer.
    NeuralNetworkOptimizer_F optimizer("./data/training/");
    // Train for 1 epoch!
    optimizer.train(digitRecognizer, 0.001);
}
