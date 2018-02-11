apt-get update -qq
./bootstrap_bolero.sh
cd bolero-dev
source env.sh
cd learning/bolero
git checkout $CIRCLE_BRANCH
${MARS_SCRIPT_DIR}/pybob.py install
