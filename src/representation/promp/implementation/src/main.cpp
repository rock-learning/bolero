#include <iostream>
#include "BasisFunctions.h"
#include <sstream>
#include "Trajectory.h"
#include <Eigen/Dense>
using namespace promp;

int main( int argc, const char* argv[] )
{
	StrokeBasisFunctions bf(30,0.05);
        for( int i = 0; i < 1; i++){
          for( double x = 0.; x < 1; x+= 1.0/1000){
            Eigen::VectorXd vec(1);
            vec << x; 
            std::cout << bf.getValue(vec)(0,i)  << std::endl;
          }
        }
	
}