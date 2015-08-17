#ifndef BOLERO_CONTROLLER_H
#define BOLERO_CONTROLLER_H

#ifdef _PRINT_HEADER_
  #warning "Controller.h"
#endif


namespace bolero {
  class Controller {
  public:
    static bool exitController;
    int run();
  }; /* end of class Controller */
} /* end of namespace bolero */


#endif /* BOLERO_CONTROLLER_H */
