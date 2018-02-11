mkdir /tmp/test_release_env
cp bolero_release.zip /tmp/test_release_env
cd /tmp/test_release_env
unzip bolero_release.zip
cd bolero-dev
source env.sh
./install.sh
cd learning/bolero
nosetests