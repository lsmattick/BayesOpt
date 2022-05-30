FROM python:3.8

# set termainal color properties
ENV TERM xterm-256color

# create virtual env
RUN python3 -m venv /benv/venv

# install requirements
COPY requirements.txt .
RUN . /benv/venv/bin/activate && \
    pip install -r requirements.txt

# update and install apps
RUN apt update && \
    apt install -y \
        zsh \
        vim

# set default to zsh
RUN chsh -s $(which zsh)

# download oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# clone powerlevel10k repo to /root/.oh-my-zsh/themes/
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git /root/.oh-my-zsh/themes/powerlevel10k

# add preset p10k configs so there is no set up required
COPY p10k_config_files/.p10k.zsh /root/.p10k.zsh
COPY p10k_config_files/.zshrc /root/.zshrc

# add vscode-server directory containing python and jupyter vs code exensions
COPY vscode-server /root/.vscode-server

# change working directory to root
WORKDIR /root

# start terminal on launch
CMD . /benv/venv/bin/activate && /bin/zsh
