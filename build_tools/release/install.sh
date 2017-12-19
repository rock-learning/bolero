#! /bin/bash

AUTOPROJ_CURRENT_ROOT=`pwd`
MARS_SCRIPT_DIR=$AUTOPROJ_CURRENT_ROOT/pybob

echo "#! /bin/sh" > env.sh
echo "export AUTOPROJ_CURRENT_ROOT=$AUTOPROJ_CURRENT_ROOT" >> env.sh
echo "export MARS_SCRIPT_DIR='$AUTOPROJ_CURRENT_ROOT/pybob'
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
" >> env.sh

echo "autoprojEnv: false" > pybob/pybob.yml
echo "buildconfAddress: https://github.com/rock-learning/bolero_buildconf.git" >> pybob/pybob.yml
echo "buildconfBranch: ''" >> pybob/pybob.yml
echo "defBuildType: debug" >> pybob/pybob.yml
echo "devDir: ${AUTOPROJ_CURRENT_ROOT}" >> pybob/pybob.yml
echo "pyScriptDir: ${AUTOPROJ_CURRENT_ROOT}/pybob" >> pybob/pybob.yml
echo "rockFlavor: master" >> pybob/pybob.yml

source env.sh
cd pybob
./pybob.py install
