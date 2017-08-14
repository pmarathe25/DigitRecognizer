BUILDDIR = ./build/
BINDIR = ./bin/
SRCDIR = ./src/
SCRIPTSDIR = ./scripts/
# Include paths
LIBMATH_INCLUDEPATH = /home/pranav/C++/Math/include/
LIBNN_INCLUDEPATH = /home/pranav/C++/NeuralNetwork/include/
LIBSTEALTHDIR_INCLUDEPATH = /home/pranav/C++/StealthDirectory/include/
INCLUDEDIR = -I./include/ -I$(LIBMATH_INCLUDEPATH) -I$(LIBNN_INCLUDEPATH) -I$(LIBSTEALTHDIR_INCLUDEPATH)
# Libraries
LIBSTEALTHMAT = /home/pranav/C++/Math/lib/libstealthmat.so
LIBSTEALTHDIR = /home/pranav/C++/StealthDirectory/lib/libstealthdir.so
LIBS = $(LIBSTEALTHMAT) $(LIBSTEALTHDIR)
# Headers
HEADERFILES += $(addprefix $(LIBMATH_INCLUDEPATH)/, StealthMatrix.hpp)
HEADERFILES += $(addprefix $(LIBNN_INCLUDEPATH)/, Layer/FullyConnectedLayer.hpp NeuralNetwork.hpp NeuralNetworkOptimizer.hpp NeuralNetworkSaver.hpp)
# Object files
CREATEMINIBATCHES_OBJS = $(addprefix $(BUILDDIR)/, createMinibatches.o)
DIGITRECOGNIZERTRAINER_OBJS = $(addprefix $(BUILDDIR)/, digitRecognizerTrainer.o)
# Compiler
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -Wno-deprecated-gpu-targets

all: $(BINDIR)/createMinibatches $(BINDIR)/digitRecognizerTrainer

$(BINDIR)/digitRecognizerTrainer: $(DIGITRECOGNIZERTRAINER_OBJS) $(LIBSTEALTHMAT)
	$(CXX) $(LFLAGS) $(LIBS) $(DIGITRECOGNIZERTRAINER_OBJS) -o $(BINDIR)/digitRecognizerTrainer

$(BUILDDIR)/digitRecognizerTrainer.o: $(HEADERFILES) $(SRCDIR)/DigitRecognizerTrainer.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/DigitRecognizerTrainer.cu -o $(BUILDDIR)/digitRecognizerTrainer.o

$(BINDIR)/createMinibatches: $(CREATEMINIBATCHES_OBJS) $(LIBSTEALTHMAT)
	$(CXX) $(LFLAGS) $(LIBSTEALTHMAT) $(CREATEMINIBATCHES_OBJS) -o $(BINDIR)/createMinibatches

$(BUILDDIR)/createMinibatches.o: $(HEADERFILES) $(SRCDIR)/CreateMinibatches.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/CreateMinibatches.cu -o $(BUILDDIR)/createMinibatches.o

clean:
	rm $(CREATEMINIBATCHES_OBJS) $(DIGITRECOGNIZERTRAINER_OBJS) $(BINDIR)/*

train: $(BINDIR)/digitRecognizerTrainer
	$(SCRIPTSDIR)/train.sh

dataset: $(BINDIR)/createMinibatches
	$(SCRIPTSDIR)/processMinibatches.sh
