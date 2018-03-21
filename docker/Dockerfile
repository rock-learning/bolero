FROM af01/bolero_dependencies

ARG install_mars_env=1

RUN apt-get update -qq

COPY docker_bootstrap_bolero.sh /opt/docker_bootstrap_bolero.sh
RUN mkdir /opt -p \
    && wget "https://raw.githubusercontent.com/rock-learning/bolero/master/bootstrap_bolero.sh" \
    && mv bootstrap_bolero.sh /opt/bootstrap_bolero.sh

RUN chmod +x /opt/bootstrap_bolero.sh && sleep 1
RUN chmod +x /opt/docker_bootstrap_bolero.sh && sleep 1

RUN /bin/bash -c "/opt/docker_bootstrap_bolero.sh ${install_mars_env}"

# For GUIs:
ENV DISPLAY ":0"
ENV QT_X11_NO_MITSHM "1"

CMD ["/bin/bash"]

