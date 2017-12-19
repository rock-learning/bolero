#! /bin/sh
export MARS_SCRIPT_DIR="$AUTOPROJ_CURRENT_ROOT/pybob"
export PATH="$PATH:$AUTOPROJ_CURRENT_ROOT/install/bin"
export LD_LIBRARY_PATH="$AUTOPROJ_CURRENT_ROOT/install/lib:$DYLD_LIBRARY_PATH"
export ROCK_CONFIGURATION_PATH="$AUTOPROJ_CURRENT_ROOT/install/configuration"
export PYTHONPATH="$AUTOPROJ_CURRENT_ROOT/install/lib/python2.7/site-packages:$PYTHONPATH"
if [ x${PKG_CONFIG_PATH} = "x" ]; then
  export PKG_CONFIG_PATH="$AUTOPROJ_CURRENT_ROOT/install/lib/pkgconfig"
else
  export PKG_CONFIG_PATH="$AUTOPROJ_CURRENT_ROOT/install/lib/pkgconfig:$PKG_CONFIG_PATH"
fi
alias bob='${MARS_SCRIPT_DIR}/pybob.py'
alias bob-bootstrap='${MARS_SCRIPT_DIR}/pybob.py bootstrap'
alias bob-install='${MARS_SCRIPT_DIR}/pybob.py install'
alias bob-rebuild='${MARS_SCRIPT_DIR}/pybob.py rebuild'
alias bob-build='${MARS_SCRIPT_DIR}/pybob.py'
alias bob-diff='${MARS_SCRIPT_DIR}/pybob.py diff'
alias bob-list='${MARS_SCRIPT_DIR}/pybob.py list'
alias bob-fetch='${MARS_SCRIPT_DIR}/pybob.py fetch'
alias bob-show-log='${MARS_SCRIPT_DIR}/pybob.py show-log'
. ${MARS_SCRIPT_DIR}/auto_complete.sh
