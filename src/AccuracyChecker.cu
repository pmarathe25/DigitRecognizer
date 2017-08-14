#include "StealthMatrix.hpp"
#include "StealthDirectory.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSaver.hpp"
#include <vector>
#include <fstream>

template <typename T>
__device__ T isNonZero(T in) {
    return in != 0;
}

// Define layers using a custom matrix class.
typedef StealthAI::SigmoidFCL<StealthMatrix_F> SigmoidFCL_F;
typedef StealthAI::LeakyReLUFCL<StealthMatrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = StealthAI::NeuralNetwork<StealthMatrix_F, Layers...>;
// Define a dataset using a custom matrix class;
typedef StealthAI::DataSet<StealthMatrix_F> DataSet_F;

int main() {
    // Create some layers.
    SigmoidFCL_F inputLayer(784, 250);
    SigmoidFCL_F hiddenLayer(250, 30);
    SigmoidFCL_F outputLayer(30, 10);
    // Create the network.
    NeuralNetwork_F<SigmoidFCL_F, SigmoidFCL_F, SigmoidFCL_F> digitRecognizer(inputLayer, hiddenLayer, outputLayer);
    // Load!
    StealthAI::NeuralNetworkSaver::load(digitRecognizer, "./network/DigitRecognizer.nn");

    // Load testing set.
    DataSet_F testingInputs;
    DataSet_F testingExpectedOutputs;
    StealthDirectory::Directory testingDir("./data/testing");
    for (auto minibatch : testingDir) {
        std::ifstream inputFile(minibatch.getPath());
        testingInputs.emplace_back(inputFile);
        testingExpectedOutputs.emplace_back(inputFile);
    }
    // Figure out total accuracy.
    int numIncorrect = 0, total = 0;
    for (int i = 0; i < testingInputs.size(); ++i) {
        StealthMatrix_F actualOutput = digitRecognizer.feedForward(testingInputs[i]);
        StealthMatrix incorrect = (actualOutput.argmax() - testingExpectedOutputs[i].argmax()).applyFunction<isNonZero>();
        numIncorrect += incorrect.weightedSum(1)[0];
        total += actualOutput.numRows();
    }
    std::cout << "Correct: " << total - numIncorrect << '\n';
    std::cout << "Incorrect: " << numIncorrect << '\n';
    std::cout << "Total: " << total << '\n';
    std::cout << "Accuracy: " << ((total - numIncorrect) / (float)(total)) * 100 << "%"  << '\n';
}
