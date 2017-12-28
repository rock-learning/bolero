#pragma once

#include <stdio.h>  /* defines FILENAME_MAX */
#include <unistd.h>
#include <fstream>
#include <iostream>
#define get_current_dir getcwd

#ifdef __WIN32__
typedef unsigned int uint;
#endif

class TestHelpers
{
  public:
  static double random_double(){
      return rand() / (double)RAND_MAX;
  }
  static int random_int(){
      return rand() % 100;
  }

  static std::vector<double> random_vector(int size){
      std::vector<double> vect(size);
      for(uint i=0; i<vect.size(); i++){
          vect[i] = random_double();
      }
      return vect;
  }

  static std::vector<double> vector(int size, double value){
    std::vector<double> vect(size);
    for(uint i=0; i<vect.size(); i++){
      vect[i] = value;
    }
    return vect;
  }

  static std::vector<std::vector<double> > random_matrix(int size_a, int size_b){
      std::vector<std::vector<double> > vect(size_a);
      for(uint i=0; i<vect.size(); i++){
          vect[i] = random_vector(size_b);
      }
      return vect;
  }

  static std::vector<std::vector<double> > matrix(int size_a, int size_b, double value){
    std::vector<std::vector<double> > vect(size_a);
    for(uint i=0; i<vect.size(); i++){
      vect[i] = vector(size_b, value);
    }
    return vect;
  }

  static void delete_file_if_exists(std::string filepath){
      std::ifstream ifile;
      ifile.open(filepath.c_str());
      if(ifile.is_open()){
          ifile.close();
          if(remove(filepath.c_str()) != 0){
              std::cerr << filepath << std::endl;
              perror("Error deleting file");
          }
      }
  }

};

