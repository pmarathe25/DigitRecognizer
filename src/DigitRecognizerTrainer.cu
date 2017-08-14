#include "StealthMatrix.hpp"
#include "StealthDirectory.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSaver.hpp"
#include "NeuralNetworkOptimizer.hpp"
#include <vector>
#include <fstream>
// Define layers using a custom matrix class.
typedef StealthAI::SigmoidFCL<StealthMatrix_F> SigmoidFCL_F;
typedef StealthAI::LeakyReLUFCL<StealthMatrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = StealthAI::NeuralNetwork<StealthMatrix_F, Layers...>;
// Define a dataset using a custom matrix class;
typedef StealthAI::DataSet<StealthMatrix_F> DataSet_F;
// Define an optimizer using a custom matrix class.
typedef StealthAI::NeuralNetworkOptimizer<StealthMatrix_F, StealthAI::mse<StealthMatrix_F>, StealthAI::mse_prime<StealthMatrix_F>> NeuralNetworkOptimizer_MSE_F;

int main() {
    // Create some layers.
    SigmoidFCL_F inputLayer(784, 250);
    SigmoidFCL_F hiddenLayer(250, 30);
    SigmoidFCL_F outputLayer(30, 10);
    // Create the network.
    NeuralNetwork_F<SigmoidFCL_F, SigmoidFCL_F, SigmoidFCL_F> digitRecognizer(inputLayer, hiddenLayer, outputLayer);
    // Create an optimizer.
    NeuralNetworkOptimizer_MSE_F optimizer;
    // Load data.
    DataSet_F trainingInputs, trainingExpectedOutputs;
    StealthDirectory::Directory dataDir("./data/training");
    for (auto minibatch : dataDir) {
        std::ifstream inputFile(minibatch.getPath());
        trainingInputs.emplace_back(inputFile);
        trainingExpectedOutputs.emplace_back(inputFile);
    }
    std::cout << "Loaded " << trainingInputs.size() << " minibatches" << '\n';
    // Train!
    optimizer.train<30>(digitRecognizer, trainingInputs, trainingExpectedOutputs, 0.01);
    // Save!
    StealthAI::NeuralNetworkSaver::save(digitRecognizer, "./network/DigitRecognizer.nn");
}
