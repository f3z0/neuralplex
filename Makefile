CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS = src/neuron.o src/neural_net.o src/test_network.o

LIBS =

TARGET = build/TestNetwork

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
