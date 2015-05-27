#pragma once
namespace behavior_learning { namespace bl_loader {

/**
* Is used by the BLLoader to manage it's loadable python libraries.
* All loadable python wrappers need to implement this Interface.
*/
class PyLoadable {
public:
  virtual ~PyLoadable() {
  }
};

}}//end namespace