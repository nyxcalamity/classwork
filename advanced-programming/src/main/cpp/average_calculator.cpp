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

/*
   Asks a user to input two values and calculates their mean value.
*/
int main(int argc, char* argv[]){
	double a,b;
	std::cout << "Please enter 1st value: " << std::endl;
	std::cin >> a;
	std::cout << "Please enter 2nd value: " << std::endl;
	std::cin >> b;
	std::cout << "Average: " << (a+b)/2 << std::endl;
	return 0;
}
