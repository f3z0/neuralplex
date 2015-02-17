CXXFLAGS =	-O2 -g -Wall -fmessage-length=0  `mysql_config --cflags` -DRAPIDJSON_HAS_STDSTRING

OBJS = src/neuron.o src/neural_net.o src/test_network.o

TARGET = build/TestNetwork

$(TARGET):	$(OBJS) 
	$(CXX) -o $(TARGET) $(OBJS) `mysql_config --libs`

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)