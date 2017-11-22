#! /bin/bash

# checking minimal dependencies on Ubuntu systems...
if [ -f /etc/lsb-release ]; then
    if [ -z `which sudo` ];
    then
        echo "sudo not available, trying to install it with 'apt-get install sudo'"
        apt-get install sudo
    fi
    if [ -z `which git` ];
    then
        echo "git not available, trying to install it with 'sudo apt-get install git'"
        sudo apt-get install git
    fi
    if [ -z `which python` ];
    then
        echo "python not available, trying to install it with 'sudo apt-get install python'"
        sudo apt-get install python python-pip
    fi
    YAML_AVAILABLE=1
    `python -c "import yaml" 2> /dev/null` || YAML_AVAILABLE=0
    if [ $YAML_AVAILABLE == 0 ];
    then
          echo "python-yaml not available, trying to install it with 'sudo apt-get install python-yaml'"
          sudo apt-get install python-yaml
    fi
fi

mkdir bolero-dev
cd bolero-dev
DEV_DIR="$( cd "$( dirname "$0" )" && pwd )"
git clone https://github.com/rock-simulation/pybob.git
cd pybob

# create default config for bolero
echo "autoprojEnv: false" > pybob.yml
echo "buildconfAddress: https://github.com/rock-learning/bolero_buildconf.git" >> pybob.yml
echo "buildconfBranch: ''" >> pybob.yml
echo "defBuildType: debug" >> pybob.yml
echo "devDir: ${DEV_DIR}" >> pybob.yml
echo "pyScriptDir: ${DEV_DIR}/pybob" >> pybob.yml
echo "rockFlavor: master" >> pybob.yml

# clone build configuration
./pybob.py buildconf

cd ..
source env.sh

# build default packages
cd pybob
./pybob.py bootstrap

echo ""
echo "To continue working with bolero in this terminal perform:"
echo ""
echo "  cd bolero-dev"
echo "  source env.sh"
echo ""
echo 'Whenever you open a new terminal to work with bolero switch into bolero-dev and source the "env.sh" again to activate the bolero install for that terminal.'
echo ""
