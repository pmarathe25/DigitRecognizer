#include "StealthMatrix.hpp"
#include "Layer/FullyConnectedLayer.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSaver.hpp"
#include "NetworkDefinition.hpp"
#include <SFML/Graphics.hpp>

int main() {
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
