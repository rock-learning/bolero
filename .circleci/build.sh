wget https://raw.githubusercontent.com/rock-learning/bolero/master/bootstrap_bolero.sh
chmod +x bootstrap_bolero.sh
./bootstrap_bolero.sh
cd bolero-dev
source env.sh
cd learning/bolero
git checkout $CIRCLE_BRANCH
bob-install
