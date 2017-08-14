#include "StealthMatrix.hpp"
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

void parseHeader(std::ifstream& dataFile, std::ifstream& labelFile, int& magicNumberData,
    int& magicNumberLabels, int& numItems, int& rows, int& cols, int numMinibatches, int& minibatchSize) {
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
    // Figure out the size of a minibatch.
    minibatchSize = std::ceil(numItems / (float) numMinibatches);
    // Output
    std::cout << "Data magic number: " << littleToBigEndian(magicNumberData) << '\n';
    std::cout << "Label magic number: " << littleToBigEndian(magicNumberLabels) << '\n';
    std::cout << "Number of Items: " << numItems << '\n';
    std::cout << "Dimensions: " << rows << "x" << cols << '\n';
    std::cout << "Minibatch Size: " << minibatchSize << '\n';
}

void parseMinibatch(std::ifstream& dataFile, std::ifstream& labelFile, StealthMatrix_UC& minibatchDataRaw, StealthMatrix_UC& minibatchLabelsRaw) {
    dataFile.read(reinterpret_cast<char*>(&minibatchDataRaw[0]), sizeof(minibatchDataRaw[0]) * minibatchDataRaw.size());
    labelFile.read(reinterpret_cast<char*>(&minibatchLabelsRaw[0]), sizeof(minibatchLabelsRaw[0]) * minibatchLabelsRaw.size());
}

void processMinibatch(const StealthMatrix_UC& minibatchDataRaw, const StealthMatrix_UC& minibatchLabelsRaw, std::string& outputPath, int minibatchNum) {
    // StealthMatrix_F minibatchData = (255 - minibatchDataRaw.asType<float>()) / 255;
    StealthMatrix_F minibatchData = (minibatchDataRaw.asType<float>()) / 255;
    StealthMatrix_F minibatchLabels(minibatchLabelsRaw.numRows(), 10);
    for (int row = 0; row < minibatchLabelsRaw.numRows(); ++row) {
        minibatchLabels.at(row, minibatchLabelsRaw[row]) = 1.0;
    }
    // Save!
    std::string minibatchSaveFile = outputPath + "/" + std::to_string(minibatchNum) + ".minibatch";
    std::cout << "Saving minibatch " << minibatchNum << " to " << minibatchSaveFile << '\n';
    std::ofstream mbFile(minibatchSaveFile);
    minibatchData.save(mbFile);
    minibatchLabels.save(mbFile);
}

// Translates the MNIST dataset into a matrix friendly format.
int main(int argc, char const *argv[]) {
    std::string dataPath, labelPath, outputPath;
    int numMinibatches = 1;
    if (argc > 3) {
        // Get the raw data file and then the labels.
        dataPath = argv[1];
        labelPath = argv[2];
        outputPath = argv[3];
        numMinibatches = (argc % 2) ? std::stoi(argv[argc - 1]) : numMinibatches;
    } else {
        std::cout << "Usage: " << argv[0] << " DATA-FILE LABELS-FILE OUTPUT-DIR [# MINIBATCHES]" << '\n';
        return 1;
    }
    // Metadata
    int magicNumberData, magicNumberLabels, numItems, rows, cols, minibatchSize;
    // Open files
    std::ifstream dataFile(dataPath, std::ios::binary);
    std::ifstream labelFile(labelPath, std::ios::binary);
    // Load into matrices.
    if (dataFile.is_open() && labelFile.is_open()) {
        parseHeader(dataFile, labelFile, magicNumberData, magicNumberLabels, numItems, rows, cols, numMinibatches, minibatchSize);
        // Data contains images, labels are 10 values with 1 of them equal to 1.0.
        StealthMatrix_UC minibatchDataRaw(minibatchSize, rows * cols), minibatchLabelsRaw(minibatchSize, 1);
        // Loop over all minibatches.
        for (int i = 0; i < numMinibatches - 1; ++i) {
            // Read in data 1 minibatch at a time.
            parseMinibatch(dataFile, labelFile, minibatchDataRaw, minibatchLabelsRaw);
            // Process and save the minibatch.
            processMinibatch(minibatchDataRaw, minibatchLabelsRaw, outputPath, i);
        }
        // Handle leftover items.
        int itemsRemaining = numItems - minibatchSize * (numMinibatches - 1);
        minibatchDataRaw = StealthMatrix_UC(itemsRemaining, rows * cols);
        minibatchLabelsRaw = StealthMatrix_UC(itemsRemaining, 1);
        // Read data and labels for last minibatch.
        parseMinibatch(dataFile, labelFile, minibatchDataRaw, minibatchLabelsRaw);
        // Process and save it.
        processMinibatch(minibatchDataRaw, minibatchLabelsRaw, outputPath, numMinibatches - 1);
    } else {
        throw std::invalid_argument("Could not open data.");
    }
}
