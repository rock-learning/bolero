#! /bin/bash

export PYTHON="${PYTHON:-python}"
if [ -z `which $PYTHON` ];
then
    echo -e "\e[31mPython '$PYTHON' not found.\e[0m"
    exit 1
fi
CYTHON_AVAILABLE=1
`$PYTHON -c "import Cython" 2> /dev/null` || CYTHON_AVAILABLE=0
if [ CYTHON_AVAILABLE == 0 ];
then
        echo "Cython for $PYTHON not available, trying to install it with '$PYTHON -m pip install Cython'"
        $PYTHON -m pip install Cython
fi
echo -e "\e[31mUsing Python: $PYTHON (located at `which $PYTHON`)\e[0m"

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
    if [ -z `which cmake` ];
    then
        echo "cmake not available, trying to install it with 'sudo apt-get install cmake'"
        sudo apt-get install cmake --yes
    fi
    if [ -z `which $PYTHON` ];
    then
        echo "$PYTHON not available, trying to install it with 'sudo apt-get install $PYTHON'"
        sudo apt-get install $PYTHON $PYTHON-pip --yes
    fi
    YAML_AVAILABLE=1
    `$PYTHON -c "import yaml" 2> /dev/null` || YAML_AVAILABLE=0
    if [ $YAML_AVAILABLE == 0 ];
    then
          echo "$PYTHON-yaml not available, trying to install it with 'sudo apt-get install $PYTHON-yaml'"
          sudo apt-get install $PYTHON-yaml --yes
    fi
    DISTRO_AVAILABLE=1
    `$PYTHON -c "import distro" 2> /dev/null` || DISTRO_AVAILABLE=0
    if [ DISTRO_AVAILABLE == 0 ];
    then
          echo "$PYTHON-distro not available, trying to install it with 'sudo apt-get install $PYTHON-distro'"
          sudo apt-get install $PYTHON-distro --yes
    fi

fi

mkdir bolero-dev
cd bolero-dev
DEV_DIR="$( cd "$( dirname "$0" )" && pwd )"

if [ -f /mingw64.exe ]; then
  DEV_DIR="$(cmd //c echo $DEV_DIR)"
fi
echo -e "\e[31mBOLeRo development directory: $DEV_DIR\e[0m"

echo -e "\e[31mDownloading pybob, BOLeRo's build manager...\e[0m"
git clone https://github.com/rock-simulation/pybob.git
echo -e "\e[31mDone.\e[0m"
cd pybob

# create default config for bolero
echo "autoprojEnv: false" > pybob.yml
echo "buildconfAddress: https://github.com/rock-learning/bolero_buildconf.git" >> pybob.yml
echo "buildconfBranch: ''" >> pybob.yml
echo "defBuildType: debug" >> pybob.yml
echo "devDir: ${DEV_DIR}" >> pybob.yml
echo "pyScriptDir: ${DEV_DIR}/pybob" >> pybob.yml
echo "rockFlavor: master" >> pybob.yml
echo "numCores: 1" >> pybob.yml

# clone build configuration
echo -e "\e[31mDownloading sources...\e[0m"
$PYTHON pybob.py buildconf
echo -e "\e[31mDone.\e[0m"

cd ..

source env.sh

# build default packages
cd pybob
echo -e "\e[31mBuilding...\e[0m"
$PYTHON pybob.py bootstrap
echo -e "\e[31mDone.\e[0m"

echo ""
echo "To continue working with bolero in this terminal perform:"
echo ""
echo "  cd bolero-dev"
echo "  source env.sh"
echo ""
echo 'Whenever you open a new terminal to work with bolero switch into bolero-dev and source the "env.sh" again to activate the bolero install for that terminal.'
echo ""
