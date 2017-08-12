#include "Matrix.hpp"
#include <iostream>
#include <fstream>
#include <string>

__device__ __host__ int littleToBigEndian(int num) {
    #ifdef LITTLE_ENDIAN
        return (num & 0xFF) << 24 | (num & 0xFF00) << 8 | ((num >> 8) & 0xFF00) | ((num >> 24) & 0xFF);
    #else
        return num;
    #endif
}

void readMinibatch(std::ifstream& dataFile, std::ifstream& labelFile, Matrix_UC& minibatchDataRaw, Matrix_UC& minibatchLabelsRaw) {
    dataFile.read(reinterpret_cast<char*>(&minibatchDataRaw[0]), sizeof(minibatchDataRaw[0]) * minibatchDataRaw.size());
    labelFile.read(reinterpret_cast<char*>(&minibatchLabelsRaw[0]), sizeof(minibatchLabelsRaw[0]) * minibatchLabelsRaw.size());
}

void parseHeader(std::ifstream& dataFile, std::ifstream& labelFile, int& magicNumberData, int& magicNumberLabels, int& numItems, int& rows, int& cols) {
    // Read magic numbers.
    dataFile.read(reinterpret_cast<char*>(&magicNumberData), sizeof magicNumberData);
    labelFile.read(reinterpret_cast<char*>(&magicNumberLabels), sizeof magicNumberLabels);
    // Figure out how much data is in the file
    dataFile.read(reinterpret_cast<char*>(&numItems), sizeof numItems);
    labelFile.read(reinterpret_cast<char*>(&numItems), sizeof numItems);
    // Get dimensions of images.
    dataFile.read(reinterpret_cast<char*>(&rows), sizeof rows);
    dataFile.read(reinterpret_cast<char*>(&cols), sizeof cols);
    // Convert everything to big endian.
    numItems = littleToBigEndian(numItems);
    rows = littleToBigEndian(rows);
    cols = littleToBigEndian(cols);
}

void processData(Matrix_F& minibatchData, Matrix_UC& minibatchDataRaw) {
    minibatchData = (255 - minibatchDataRaw.asType<float>()) / 255;
}

void processLabels(Matrix_F& minibatchLabels, Matrix_UC& minibatchLabelsRaw) {
    for (int row = 0; row < minibatchLabelsRaw.numRows(); ++row) {
        minibatchLabels.at(row, minibatchLabelsRaw[row]) = 1.0;
    }
}

// Translates the MNIST dataset into a matrix friendly format.
int main(int argc, char const *argv[]) {
    std::string dataPath, labelPath;
    int numMinibatches = 1;
    try {
        // Get the raw data file and then the labels.
        dataPath = argv[1];
        labelPath = argv[2];
        numMinibatches = (argc % 2) ? numMinibatches : std::stoi(argv[argc - 1]);
    } catch (const std::exception& e) {
        std::cout << "Usage: " << argv[0] << " DATA LABELS [# MINIBATCHES]" << '\n';
        return 1;
    }
    // Two matrices contain data and labels respectively for a minibatch.
    // Load data into matrices to write out to batches.
    int magicNumberData, magicNumberLabels, numItems, rows, cols;
    // Open files
    std::ifstream dataFile(dataPath, std::ios::binary);
    std::ifstream labelFile(labelPath, std::ios::binary);
    // Load into matrices.
    if (dataFile.is_open() && labelFile.is_open()) {
        parseHeader(dataFile, labelFile, magicNumberData, magicNumberLabels, numItems, rows, cols);
        // Figure out the size of a minibatch.
        int minibatchSize = std::ceil(numItems / (float) numMinibatches);
        // Data contains images, labels are 10 values with 1 of them equal to 1.
        Matrix_UC minibatchDataRaw(minibatchSize, rows * cols), minibatchLabelsRaw(minibatchSize, 1);
        Matrix_F minibatchData(minibatchSize, rows * cols), minibatchLabels(minibatchSize, 10);

        std::cout << "Data magic number: " << littleToBigEndian(magicNumberData) << '\n';
        std::cout << "Label magic number: " << littleToBigEndian(magicNumberLabels) << '\n';
        std::cout << "Num Items: " << numItems << '\n';
        std::cout << "Dimensions: " << rows << "x" << cols << '\n';
        std::cout << "Number of Minibatches: " << numMinibatches << '\n';
        std::cout << "Minibatch Size: " << minibatchSize << '\n';

        // Loop over all minibatches.
        for (int i = 0; i < numMinibatches - 1; ++i) {
            // Read in data 1 minibatch at a time.
            readMinibatch(dataFile, labelFile, minibatchDataRaw, minibatchLabelsRaw);
            // TODO: Process and save the minibatch.
            processData(minibatchData, minibatchDataRaw);
            processLabels(minibatchLabels, minibatchLabelsRaw);

        }

        // Handle leftover items.
        int itemsRemaining = numItems - minibatchSize * (numMinibatches - 1);
        minibatchDataRaw = Matrix_UC(itemsRemaining, rows * cols);
        minibatchLabelsRaw = Matrix_UC(itemsRemaining, 1);
        minibatchData = Matrix_F(itemsRemaining, rows * cols);
        minibatchLabels = Matrix_F(itemsRemaining, 10);
        // Read data and labels for last minibatch.
        readMinibatch(dataFile, labelFile, minibatchDataRaw, minibatchLabelsRaw);
        // Process it.
        processData(minibatchData, minibatchDataRaw);
        processLabels(minibatchLabels, minibatchLabelsRaw);

        std::cout << "Dimensions: " << rows << "x" << cols << '\n';
        std::cout << "Items Remaining: " << itemsRemaining << '\n';
        minibatchData.reshape(28 * minibatchData.numRows()).display("Minibatch of 1 image.");
        minibatchLabels.display("Labels");

    }
}
