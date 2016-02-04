#include <BLLoader.h>
#include <LoadableBehavior.h>
#include <iostream>
#include <sstream>


void printPose(double* pose)
{
  std::cout << "Pos. " << pose[0] << ", " << pose[1] << ", " << pose[2]
    << std::endl << "Rot. " << pose[9] << ", " << pose[10]
    << ", " << pose[11] << ", " << pose[12] << std::endl << std::endl;
}


void setStartToCurrentPose(bolero::LoadableBehavior* behav, double* pose)
{
  std::stringstream newStartPose;
  newStartPose << "startPosition: [" << pose[0] << ", " << pose[1]
    << ", " << pose[2] << "]" << std::endl;
  newStartPose << "startRotation: [" << pose[9] << ", " << pose[10]
    << ", " << pose[11] << ", " << pose[12] << "]" << std::endl;
  behav->configureYaml(newStartPose.str());
}


int main()
{

  bolero::bl_loader::BLLoader loader;
  loader.loadConfigFile("load_libraries.txt");

  //the pointer is managed by libManager. No need to use auto_ptr
  bolero::LoadableBehavior* behav = loader.acquireBehavior("RigidBodyDmp");

  behav->initialize("model.yaml");
  behav->configure("config.yaml");

  std::string newGoalPose = "endPosition: [0.3, -0.4, 0.2]\n"
                            "endRotation: [0.27216553, 0.40824829, 0.54433105, 0.68041382]";

  // initial position and rotation are different from original configuration
  double data[13] = {
    -0.5, -0.5, 0.5,
    0.09067497083437709, -0.018982788402122086, -0.0017532559390687208,
    1.1358714507546386, 0.5246758316059669, -1.2672900153373745,
    0.18257419, 0.36514837, 0.54772256, 0.73029674};

  printPose(data);
  for(int i = 0; behav->canStep(); i++)
  {
    if(i == 0)
      setStartToCurrentPose(behav, data);
    else if(i == 10)
      behav->configureYaml(newGoalPose);

    behav->setInputs(&data[0], 13);
    behav->step();
    behav->getOutputs(&data[0], 13);
    printPose(data);
  }

  loader.releaseLibrary("RigidBodyDmp");
}

