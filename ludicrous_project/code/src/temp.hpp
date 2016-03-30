//#include "helper.h"
#include <iostream>

class Temp {
private:

public:
    Temp() {
        std::cout << "constructor called: "  << std::endl;
        //printSomething();
    }

    ~Temp() {

    }

    void printSomething() {
        std::cout << "printing something  :O" << std::endl;
    }

};
