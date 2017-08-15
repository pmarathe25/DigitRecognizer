#ifndef NETWORK_DEFINITION_H
#define NETWORK_DEFINITION_H
#include "StealthMatrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
// Define layers using a custom matrix class.
typedef StealthAI::SigmoidFCL<StealthMatrix_F> SigmoidFCL_F;
typedef StealthAI::LeakyReLUFCL<StealthMatrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = StealthAI::NeuralNetwork<StealthMatrix_F, Layers...>;

// Create some layers.
SigmoidFCL_F inputLayer(784, 392);
SigmoidFCL_F hiddenLayer1(392, 240);
SigmoidFCL_F hiddenLayer2(240, 120);
SigmoidFCL_F hiddenLayer3(120, 30);
SigmoidFCL_F outputLayer(30, 10);
// Create the network.
NeuralNetwork_F<SigmoidFCL_F, SigmoidFCL_F, SigmoidFCL_F, SigmoidFCL_F, SigmoidFCL_F> digitRecognizer(inputLayer, hiddenLayer1, hiddenLayer2, hiddenLayer3, outputLayer);

#endif
