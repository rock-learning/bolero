echo "Installation of Bolero in Docker container"

INSTALLMARSENV=$1
WORK=/opt
echo "Working folder:" $WORK

[ $INSTALLMARSENV -eq 1 ] && echo "Optional MARS environment is going to be installed" || echo "Optional MARS environment is NOT going to be installed"
cd $WORK

./bootstrap_bolero.sh

cd $WORK
cd bolero-dev
source env.sh

if [ $INSTALLMARSENV -eq 1 ]; then
	echo "Enabling optional MARS environment"
	sed -i -E 's/# (- .+\/mars_environment)/\1/' autoproj/manifest
	sed -i -E '/ - .+\/mars_environment/a\ \ - simulation/mars/entity_generation/smurf' autoproj/manifest
	sed -i -E '/ - .+\/mars_environment/a\ \ - simulation/mars/smurf_loader' autoproj/manifest
	echo "Running bootstrap again"
	cd pybob
	./pybob.py bootstrap
	cd $WORK/bolero-dev
	source env.sh
else
	echo "Not installing optional MARS environment"
fi

cd $WORK

echo "cd /opt/bolero-dev && source env.sh && cd ~" >> ~/.bashrc

