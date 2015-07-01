#include <BLLoader.h>
#include <LoadableBehavior.h>
#include <iostream>


int main()
{

  bolero::bl_loader::BLLoader loader;
  loader.loadConfigFile("load_libraries.txt");

  //the pointer is managed by libManager. No need to use auto_ptr
  bolero::LoadableBehavior* behav = loader.acquireBehavior("QuaternionDmpBehavior");

  behav->initialize("model.yaml");
  behav->configure("config.yaml");

  double data[4] = {0.1, 0.2, 0.3, 0.4};

  std::cout << data[0] << " " << data[1] << " " << data[2] << " " <<
               data[3]  << std::endl;
  while(behav->canStep())
  {
    behav->setInputs(&data[0], 6);
    behav->step();
    behav->getOutputs(&data[0], 6);
    std::cout << data[0] << " " << data[1] << " " << data[2] << " " <<
                 data[3] << std::endl;
  }


  loader.releaseLibrary("QuaternionDmpBehavior");
}


