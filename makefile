BUILDDIR = ./build/
BINDIR = ./bin/
LIBMATRIX = /home/pranav/C++/Math/lib/libmatrix.so
LIBMATH_INCLUDEPATH = /home/pranav/C++/Math/include/
LIBNN_INCLUDEPATH = /home/pranav/C++/NeuralNetwork/include/
LIBS = $(LIBMATRIX)
INCLUDEDIR = -I./include/ -I$(LIBMATH_INCLUDEPATH) -I$(LIBNN_INCLUDEPATH)
HEADERFILES += $(addprefix $(LIBMATH_INCLUDEPATH)/, Matrix.hpp)
HEADERFILES += $(addprefix $(LIBNN_INCLUDEPATH)/, Layer/FullyConnectedLayer.hpp NeuralNetwork.hpp NeuralNetworkOptimizer.hpp NeuralNetworkSaver.hpp)
CREATEMINIBATCHES_OBJS = $(addprefix $(BUILDDIR)/, createMinibatches.o)
DIGITRECOGNIZERTRAINER_OBJS = $(addprefix $(BUILDDIR)/, digitRecognizerTrainer.o)
SRCDIR = ./src/
SCRIPTSDIR = ./scripts/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -Wno-deprecated-gpu-targets

all: $(BINDIR)/createMinibatches $(BINDIR)/digitRecognizerTrainer

$(BINDIR)/digitRecognizerTrainer: $(DIGITRECOGNIZERTRAINER_OBJS) $(LIBMATRIX)
	$(CXX) $(LFLAGS) $(LIBMATRIX) $(DIGITRECOGNIZERTRAINER_OBJS) -o $(BINDIR)/digitRecognizerTrainer

$(BUILDDIR)/digitRecognizerTrainer.o: $(HEADERFILES) $(SRCDIR)/DigitRecognizerTrainer.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/DigitRecognizerTrainer.cu -o $(BUILDDIR)/digitRecognizerTrainer.o

$(BINDIR)/createMinibatches: $(CREATEMINIBATCHES_OBJS) $(LIBMATRIX)
	$(CXX) $(LFLAGS) $(LIBMATRIX) $(CREATEMINIBATCHES_OBJS) -o $(BINDIR)/createMinibatches

$(BUILDDIR)/createMinibatches.o: $(HEADERFILES) $(SRCDIR)/CreateMinibatches.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/CreateMinibatches.cu -o $(BUILDDIR)/createMinibatches.o

clean:
	rm $(CREATEMINIBATCHES_OBJS) $(DIGITRECOGNIZERTRAINER_OBJS) $(BINDIR)/*

train: $(BINDIR)/digitRecognizerTrainer
	$(SCRIPTSDIR)/train.sh

process: $(BINDIR)/createMinibatches
	$(SCRIPTSDIR)/processMinibatches.sh
