#! /bin/bash

# clean up environment variables that might interfere with this script
unset AUTOPROJ_CURRENT_ROOT
unset MARS_SCRIPT_DIR
unset LD_LIBRARY_PATH
unset ROCK_CONFIGURATION_PATH
unset PYTHONPATH
unset PKG_CONFIG_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# checking minimal dependencies on Ubuntu systems...
if [ -f /etc/lsb-release ]; then
    if [ -z `which sudo` ];
    then
        echo "sudo not available, trying to install it with 'apt-get install sudo'"
        apt-get install sudo --yes
    fi
    if [ -z `which git` ];
    then
        echo "git not available, trying to install it with 'sudo apt-get install git'"
        sudo apt-get install git --yes
    fi
    if [ -z `which unzip` ];
    then
        echo "unzip not available, trying to install it with 'sudo apt-get install unzip'"
        sudo apt-get install unzip --yes
    fi
    if [ -z `which python` ];
    then
        echo "python not available, trying to install it with 'sudo apt-get install python'"
        sudo apt-get install python python-pip --yes
    fi
    YAML_AVAILABLE=1
    `python -c "import yaml" 2> /dev/null` || YAML_AVAILABLE=0
    if [ $YAML_AVAILABLE == 0 ];
    then
          echo "python-yaml not available, trying to install it with 'sudo apt-get install python-yaml'"
          sudo apt-get install python-yaml --yes
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
cp ../../manifest ../autoproj/manifest
./pybob.py fetch
