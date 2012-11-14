# Builds the project
BUILD = ./build
SRC_CPP = ./src/main/cpp

all: clean ws1 ws2
	mkdir -p ~/bin/
	cp $(BUILD)/* ~/bin/

ws1:
	 g++ -Wall $(SRC_CPP)/average_calculator.cpp -o $(BUILD)/average_calculator

ws2:
	 g++ -Wall $(SRC_CPP)/simple_calculator.cpp -o $(BUILD)/simple_calculator

clean:
	mkdir -p ${BUILD}

distclean:
	#todo
