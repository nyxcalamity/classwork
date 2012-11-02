#include <iostream>

int main(int argc, char* argv[]){
	double a,b;
	std::cout << "Please enter 1st value: " << std::endl;
	std::cin >> a;
	std::cout << "Please enter 2nd value: " << std::endl;
	std::cin >> b;
	std::cout << "Average: " << (a+b)/2 << std::endl;
	return 0;
}
