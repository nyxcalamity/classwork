/*
   Copyright 2012 Denys Sobchyshak

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <iostream>
#include <string>
#include <cmath>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

/*
   Represents a calculator with various functions.
*/
class Calc {
public:
	//arithmetic functions
	static double sum(double a, double b){ return a+b; };
	static double subtract(double a, double b){ return a-b;};
	static double multiply(double a, double b){ return a*b; };
	static double divide(double a, double b){ if (b == 0) return b; return a/b; };
	static double avg(double a, double b){ return (a+b)/2; };

	//trigonometric funcitons
	static double sin(double x){ return std::sin(M_PI*x); };
	static double cos(double x){ return std::cos(M_PI*x); };
	static double tan(double x){ return std::tan(M_PI*x); };
};

void getValue(double *a){
	std::cout << "Please, enter a value: " << std::endl;
        std::cin >> (*a);
}

void getValues(double *a, double *b){
	std::cout << "Please, enter two values: " << std::endl;
        std::cin >> (*a) >> (*b);
}

bool getCommand(std::string *cmd){
	std::cout << "Please, enter a command (+,-,*,/,avg, sin, cos, tan, exit): ";
	std::cin >> (*cmd);
	if ( (*cmd) == "exit"){
		return false;
	} else {
		return true;
	}
}

int main (){
	std::string cmd;
	double a,b;

	while (getCommand(&cmd)){
		if (cmd == "sin" || cmd == "cos" || cmd == "tan"){
			getValue(&a);
			if (cmd == "sin"){
				std::cout << "Sine: " << Calc::sin(a) << std::endl;
			} else if (cmd == "cos"){
				std::cout << "Cosine: " << Calc::cos(a) << std::endl;
			} else if (cmd == "tan"){
				std::cout << "Tangent: " << Calc::tan(a) << std::endl;
			}
		} else {
			getValues(&a,&b);
			if (cmd == "+"){
				std::cout << "Summation: " << Calc::sum(a,b) << std::endl;
			} else if (cmd == "-"){
				std::cout << "Subtracttion: " << Calc::subtract(a,b) << std::endl;
			} else if (cmd == "*"){
				std::cout << "Multiplication: " << Calc::multiply(a,b) << std::endl;
			} else if (cmd == "/"){
				std::cout << "Division: " << Calc::divide(a,b) << std::endl;
			} else if (cmd == "avg"){
				std::cout << "Average: " << Calc::avg(a,b) << std::endl;
			}
		}
	}
	return 0;
}
