#include "StealthMatrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSaver.hpp"
#include <SFML/Graphics.hpp>
// Define layers using a custom matrix class.
typedef StealthAI::SigmoidFCL<StealthMatrix_F> SigmoidFCL_F;
typedef StealthAI::LeakyReLUFCL<StealthMatrix_F> LeakyReLUFCL_F;
// Define a network using a custom matrix class.
template <typename... Layers>
using NeuralNetwork_F = StealthAI::NeuralNetwork<StealthMatrix_F, Layers...>;

int main() {
    // Create some layers.
    SigmoidFCL_F inputLayer(784, 250);
    SigmoidFCL_F hiddenLayer(250, 30);
    SigmoidFCL_F outputLayer(30, 10);
    // Create the network.
    NeuralNetwork_F<SigmoidFCL_F, SigmoidFCL_F, SigmoidFCL_F> digitRecognizer(inputLayer, hiddenLayer, outputLayer);
    // Load!
    StealthAI::NeuralNetworkSaver::load(digitRecognizer, "./network/DigitRecognizer.nn");
    // Load an image too.
    sf::Image digit;
    digit.loadFromFile("./digit.png");
    int width = digit.getSize().x, height = digit.getSize().y;
    StealthMatrix_F input(width, height);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            sf::Color pixel = digit.getPixel(j, i);
            input.at(i, j) = 1.0 - (pixel.r + pixel.g + pixel.b) / (float)(3 * 255);
        }
    }
    // input.transpose().display();
    std::cout << "Loaded an image of dimensions " << width << "x" << height << '\n';
    digitRecognizer.feedForward(input.reshape(1)).argmax().display("Prediction");
    digitRecognizer.feedForward(input.reshape(1)).display("Raw Prediction");

}
