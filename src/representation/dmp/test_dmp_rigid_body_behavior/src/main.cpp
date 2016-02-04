#include <BLLoader.h>
#include <LoadableBehavior.h>
#include <iostream>


void printPose(double* data)
{
  std::cout << "Pos. " << data[0] << ", " << data[1] << ", " << data[2]
    << std::endl << "Rot. " << data[9] << ", " << data[10]
    << ", " << data[11] << ", " << data[12] << std::endl << std::endl;
}


int main()
{

  bolero::bl_loader::BLLoader loader;
  loader.loadConfigFile("load_libraries.txt");

  //the pointer is managed by libManager. No need to use auto_ptr
  bolero::LoadableBehavior* behav = loader.acquireBehavior("RigidBodyDmp");

  behav->initialize("model.yaml");
  behav->configure("config.yaml");

  std::string newConfig = "endPosition: [0.3, -0.4, 0.2]\n"
                          "endRotation: [0.18257419, 0.36514837, 0.54772256, 0.73029674]";

  double data[13] = {
    -0.222118008304239, -0.2662358159780513, 0.6547806356235144,
    0.09067497083437709, -0.018982788402122086, -0.0017532559390687208,
    1.1358714507546386, 0.5246758316059669, -1.2672900153373745,
    0.8455458276450503, 0.35979028320593454, -0.38358503075862216, -0.09200939974017508};

  printPose(data);
  for(int i = 0; behav->canStep(); i++)
  {
    if(i == 10)
      behav->configureYaml(newConfig);
    behav->setInputs(&data[0], 13);
    behav->step();
    behav->getOutputs(&data[0], 13);
    printPose(data);
  }

  loader.releaseLibrary("RigidBodyDmp");
}

