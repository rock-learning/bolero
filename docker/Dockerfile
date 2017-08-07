FROM hwiese/ubuntu-apt-fast:xenial

ARG install_mars_env=1

RUN apt-fast update -qq \
	&& apt-fast install -y \
		python-pip python-numpy python-yaml python-tk python-opencv python-colorama python-termcolor python-dev \
		libboost-all-dev libeigen3-dev libqt4-dev libtinyxml-dev libminizip-dev \
		cmake git vim net-tools inetutils-ping inetutils-traceroute wget curl sudo

RUN pip install --upgrade pip \
	&& pip install --upgrade scikit-learn \
	&& pip install --upgrade scipy \
	&& pip install --upgrade joblib \
	&& pip install --upgrade pandas \
	&& pip install --upgrade bs4 \
	&& pip install --upgrade config \
	&& pip install --upgrade matplotlib \
	&& pip install --upgrade nose \
	&& pip install --upgrade lxml

COPY docker_bootstrap_bolero.sh /opt/docker_bootstrap_bolero.sh
RUN mkdir /opt -p \
	&& curl 'https://raw.githubusercontent.com/rock-learning/bolero/master/bootstrap_bolero.sh' > /opt/bootstrap_bolero.sh

RUN chmod +x /opt/bootstrap_bolero.sh && sleep 1
RUN chmod +x /opt/docker_bootstrap_bolero.sh && sleep 1

RUN /bin/bash -c "/opt/docker_bootstrap_bolero.sh ${install_mars_env}"

# For GUIs:
ENV DISPLAY ":0"
ENV QT_X11_NO_MITSHM "1"

CMD ["/bin/bash"]

