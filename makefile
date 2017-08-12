BUILDDIR = ./build/
BINDIR = ./bin/
LIBMATRIX = /home/pranav/C++/Math/lib/libmatrix.so
LIBMATH_INCLUDEPATH = /home/pranav/C++/Math/include/
LIBNN_INCLUDEPATH = /home/pranav/C++/NeuralNetwork/include/
LIBS = $(LIBMATRIX)
INCLUDEDIR = -I./include/ -I$(LIBMATH_INCLUDEPATH) -I$(LIBNN_INCLUDEPATH)
CREATEMINIBATCHES_OBJS = $(addprefix $(BUILDDIR)/, createMinibatches.o)
SRCDIR = ./src/
TESTDIR = ./test/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -Wno-deprecated-gpu-targets

$(BINDIR)/createMinibatches: $(CREATEMINIBATCHES_OBJS) $(LIBMATRIX)
	$(CXX) $(LFLAGS) $(LIBMATRIX) $(CREATEMINIBATCHES_OBJS) -o $(BINDIR)/createMinibatches

$(BUILDDIR)/createMinibatches.o: $(LIBMATH_INCLUDEPATH)/Matrix.hpp $(SRCDIR)/CreateMinibatches.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/CreateMinibatches.cu -o $(BUILDDIR)/createMinibatches.o

clean:
	rm $(CREATEMINIBATCHES_OBJS) $(BINDIR)/*

test: $(BINDIR)/createMinibatches
	$(TESTDIR)/test.sh
