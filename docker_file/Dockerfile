FROM nvcr.io/nvidia/pytorch:21.12-py3
RUN apt update
ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install tzdata keyboard-configuration -y
RUN apt-get install build-essential git gedit fish -y
#RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#ENV PATH="/home/linuxbrew/.linuxbrew/bin:$PATH"
#ENV HOMEBREW_NO_INSTALL_CLEANUP=1
#RUN brew install graphviz
RUN apt-get install -y graphviz
RUN pip install torchviz
RUN pip install hiddenlayer
CMD ["fish"]
