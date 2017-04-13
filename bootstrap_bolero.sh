#! /bin/bash

mkdir bolero-dev
cd bolero-dev
DEV_DIR="$( cd "$( dirname "$0" )" && pwd )"
git clone https://github.com/rock-simulation/pybob.git
cd pybob

# create default config for bolero
echo "autoprojEnv: false" > pybob.yml
echo "buildconfAddress: git@git.hb.dfki.de:team-learning/bolero_buildconf.git" >> pybob.yml
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
