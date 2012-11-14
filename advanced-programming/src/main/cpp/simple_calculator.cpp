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
using namespace std;

class Calculator {
public:
	// Static methods
	static int sum(int a, int b){ return a+b; };
	static int subtract(int a, int b){ return a-b;};
	static int multiply(int a, int b){ return a*b; };
	static int divide(int a, int b){ if (b == 0) return b; return a/b; };
	static int avg(int a, int b){ return (a+b)/2; };
	//sin, cos, tan
};

void getValues(int *a, int *b){
	cout << "Please enter two values: " << endl;
        cin >> (*a) >> (*b);
}

int main (){
	int a,b;
	getValues(&a, &b);
	cout << "Summation: " << Calculator::sum(a,b) << endl;
	getValues(&a, &b);
        cout << "Subtracttion: " << Calculator::subtract(a,b) << endl;
	getValues(&a, &b);
	cout << "Multiplication: " << Calculator::multiply(a,b) << endl;
	getValues(&a, &b);
	cout << "Division: " << Calculator::divide(a,b) << endl;
	getValues(&a, &b);
	cout << "Average: " << Calculator::avg(a,b) << endl;
	return 0;
}
