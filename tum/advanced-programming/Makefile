# Builds the project
BUILD = ./build
SRC_CPP = ./src

all: init clean ha1 ha2
	cp $(BUILD)/* ~/bin/

ha1:
	 g++ -Wall $(SRC_CPP)/average_calculator.cpp -o $(BUILD)/average_calculator

ha2:
	 g++ -Wall -Wextra $(SRC_CPP)/simple_calculator.cpp -o $(BUILD)/simple_calculator

clean:
	rm -rf $(BUILD)/*

init:
	mkdir -p ${BUILD} ~/bin
