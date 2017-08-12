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

void saveMinibatch(Matrix_F& minibatchData, Matrix_F& minibatchLabels, std::string& outputPath, int minibatchNum) {
    std::string minibatchSaveFile = outputPath + "/" + std::to_string(minibatchNum) + ".minibatch";
    std::cout << "Saving minibatch " << minibatchNum << " to " << minibatchSaveFile << '\n';
    // Open a file for saving.
    std::ofstream outputFile(minibatchSaveFile, std::ios::binary);
    // Save!
    minibatchData.save(outputFile);
    minibatchLabels.save(outputFile);
    outputFile.close();

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

void parseData(std::ifstream& dataFile, std::ifstream& labelFile, Matrix_UC& minibatchDataRaw, Matrix_UC& minibatchLabelsRaw) {
    dataFile.read(reinterpret_cast<char*>(&minibatchDataRaw[0]), sizeof(minibatchDataRaw[0]) * minibatchDataRaw.size());
    labelFile.read(reinterpret_cast<char*>(&minibatchLabelsRaw[0]), sizeof(minibatchLabelsRaw[0]) * minibatchLabelsRaw.size());
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
    std::string dataPath, labelPath, outputPath;
    int numMinibatches = 1;
    try {
        // Get the raw data file and then the labels.
        dataPath = argv[1];
        labelPath = argv[2];
        outputPath = argv[3];
        numMinibatches = (argc % 2) ? std::stoi(argv[argc - 1]) : numMinibatches;
    } catch (const std::exception& e) {
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
        Matrix_UC minibatchDataRaw(minibatchSize, rows * cols), minibatchLabelsRaw(minibatchSize, 1);
        Matrix_F minibatchData(minibatchSize, rows * cols), minibatchLabels(minibatchSize, 10);
        // Loop over all minibatches.
        for (int i = 0; i < numMinibatches - 1; ++i) {
            // Read in data 1 minibatch at a time.
            parseData(dataFile, labelFile, minibatchDataRaw, minibatchLabelsRaw);
            // Process the minibatch.
            processData(minibatchData, minibatchDataRaw);
            processLabels(minibatchLabels, minibatchLabelsRaw);
            // Save.
            saveMinibatch(minibatchData, minibatchLabels, outputPath, i);
        }
        // Handle leftover items.
        int itemsRemaining = numItems - minibatchSize * (numMinibatches - 1);
        minibatchDataRaw = Matrix_UC(itemsRemaining, rows * cols);
        minibatchLabelsRaw = Matrix_UC(itemsRemaining, 1);
        minibatchData = Matrix_F(itemsRemaining, rows * cols);
        minibatchLabels = Matrix_F(itemsRemaining, 10);
        // Read data and labels for last minibatch.
        parseData(dataFile, labelFile, minibatchDataRaw, minibatchLabelsRaw);
        // Process it.
        processData(minibatchData, minibatchDataRaw);
        processLabels(minibatchLabels, minibatchLabelsRaw);
        // Save.
        saveMinibatch(minibatchData, minibatchLabels, outputPath, numMinibatches - 1);
    } else {
        throw std::invalid_argument("Could not open data.");
    }
}
