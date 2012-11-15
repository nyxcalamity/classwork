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

#define _USE_MATH_DEFINES

using namespace std;

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
	static double sin(double x){ return sin(M_PI*x); };
	static double cos(double x){ return cos(M_PI*x); };
	static double tan(double x){ return tan(M_PI*x); };
};

void getValue(double *a){
	cout << "Please, enter a value: " << endl;
        cin >> (*a);
}

void getValues(double *a, double *b){
	cout << "Please, enter two values: " << endl;
        cin >> (*a) >> (*b);
}

bool getCommand(string *cmd){
	cout << "Please, enter a command (+,-,*,/,avg, sin, cos, tan, exit): ";
	cin >> (*cmd);
	if ( (*cmd) == "exit"){
		return false;
	} else {
		return true;
	}
}

int main (){
	string cmd;
	double a,b;

	while (getCommand(&cmd)){
		if (cmd == "sin" || cmd == "cos" || cmd == "tan"){
			getValue(&a);
			if (cmd == "sin"){
				cout << "Sine: " << Calc::sin(a) << endl;
			} else if (cmd == "cos"){
				cout << "Cosine: " << Calc::cos(a) << endl;
			} else if (cmd == "tan"){
				cout << "Tangent: " << Calc::tan(a) << endl;
			}
		} else {
			getValues(&a,&b);
			if (cmd == "+"){
				cout << "Summation: " << Calc::sum(a,b) << endl;
			} else if (cmd == "-"){
				cout << "Subtracttion: " << Calc::subtract(a,b) << endl;
			} else if (cmd == "*"){
				cout << "Multiplication: " << Calc::multiply(a,b) << endl;
			} else if (cmd == "/"){
				cout << "Division: " << Calc::divide(a,b) << endl;
			} else if (cmd == "avg"){
				cout << "Average: " << Calc::avg(a,b) << endl;
			}
		}
	}
	return 0;
}
