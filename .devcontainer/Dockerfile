FROM mcr.microsoft.com/devcontainers/base:debian

COPY environment.yml /tmp/

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends vim git-lfs texlive-full \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*