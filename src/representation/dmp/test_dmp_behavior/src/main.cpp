#include <BLLoader.h>
#include <LoadableBehavior.h>
#include <iostream>


int main()
{

  bolero::bl_loader::BLLoader loader;
  loader.loadConfigFile("load_libraries.txt");

  //the pointer is managed by libManager. No need to use auto_ptr
  bolero::LoadableBehavior* behav = loader.acquireBehavior("DmpBehavior");

  behav->initialize("model.yaml");
  behav->configure("config.yaml");

  std::string newConfig = "dmp_endPosition: [11, 11]";

  double data[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  std::cout << data[0] << " " << data[1] << " " << data[2] << " " <<
               data[3] << " " << data[4] << " " << data[5] << std::endl;
  for(int i = 0; behav->canStep(); i++)
  {
    if(i == 10)
      behav->configureYaml(newConfig);
    behav->setInputs(&data[0], 6);
    behav->step();
    behav->getOutputs(&data[0], 6);
    std::cout << data[0] << " " << data[1] << " " << data[2] << " " <<
                 data[3] << " " << data[4] << " " << data[5] << std::endl;
  }


  loader.releaseLibrary("DmpBehavior");
}


