#include "StealthMatrix.hpp"
#include "StealthDirectory.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSaver.hpp"
#include "NeuralNetworkOptimizer.hpp"
#include <vector>
#include <fstream>

template <typename Matrix>
inline Matrix mseUnique(const Matrix& networkOutput, const Matrix& expectedOutput) {
    // Ensure that only one of the outputs goes high.
    return ((expectedOutput - networkOutput).pow(2) / 2).addVector((networkOutput.weightedSum(0) - 1) / 10);
}

template <typename Matrix>
inline Matrix mseUnique_prime(const Matrix& networkOutput, const Matrix& expectedOutput) {
    return networkOutput - expectedOutput.addVector((networkOutput.weightedSum(0) - 1) / 10);
}

// Define layers using a custom matrix class.
typedef StealthAI::SigmoidFCL<StealthMatrix_F> SigmoidFCL_F;
typedef StealthAI::LeakyReLUFCL<StealthMatrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = StealthAI::NeuralNetwork<StealthMatrix_F, Layers...>;
// Define a dataset using a custom matrix class;
typedef StealthAI::DataSet<StealthMatrix_F> DataSet_F;
// Define an optimizer using a custom matrix class.
typedef StealthAI::NeuralNetworkOptimizer<StealthMatrix_F, mseUnique<StealthMatrix_F>, mseUnique_prime<StealthMatrix_F>> NeuralNetworkOptimizer_MSEUNIQUE_F;

int main() {
    // Create some layers.
    SigmoidFCL_F inputLayer(784, 30);
    SigmoidFCL_F outputLayer(30, 10);
    // Create the network.
    NeuralNetwork_F<SigmoidFCL_F, SigmoidFCL_F> digitRecognizer(inputLayer, outputLayer);
    // Create an optimizer.
    NeuralNetworkOptimizer_MSEUNIQUE_F optimizer;
    // Load data.
    DataSet_F trainingInputs;
    DataSet_F trainingExpectedOutputs;
    StealthDirectory::Directory dataDir("./data/training");
    for (auto minibatch : dataDir) {
        std::ifstream inputFile(minibatch.getPath());
        trainingInputs.emplace_back(inputFile);
        trainingExpectedOutputs.emplace_back(inputFile);
    }
    std::cout << "Loaded " << trainingInputs.size() << " minibatches" << '\n';
    // Minibatch 1
    trainingExpectedOutputs[0].argmax().transpose().display("Expected Output");
    digitRecognizer.feedForward(trainingInputs[0]).argmax().transpose().display("Actual Output");
    // Train for 1 epoch!
    optimizer.getAverageCost(digitRecognizer, trainingInputs, trainingExpectedOutputs).display("Average Cost Before");
    optimizer.train<40>(digitRecognizer, trainingInputs, trainingExpectedOutputs, 0.01);
    optimizer.getAverageCost(digitRecognizer, trainingInputs, trainingExpectedOutputs).display("Average Cost After");
    // Save!
    StealthAI::NeuralNetworkSaver::save(digitRecognizer, "./network/DigitRecognizer.nn");
    // After training
    StealthMatrix_F expectedOutput = trainingExpectedOutputs[0].argmax().transpose();
    StealthMatrix_F actualOutput = digitRecognizer.feedForward(trainingInputs[0]).argmax().transpose();
    expectedOutput.display("Expected Output");
    actualOutput.display("Actual Output");
    (actualOutput - expectedOutput).display("Delta");
}
