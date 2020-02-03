# syntax=docker/dockerfile:1.0.0-experimental
# (experimental ssh forwarding: https://medium.com/@tonistiigi/build-secrets-and-ssh-forwarding-in-docker-18-09-ae8161d066)

# BASE has to have Python 3.7+ 
ARG BASE
FROM ${BASE}
WORKDIR /app

ENV GIT_SSH_COMMAND 'ssh -o "StrictHostKeyChecking no"'

RUN pip --no-cache-dir install matplotlib

ARG AGENTS_REV
RUN --mount=type=ssh git clone git@github.com:rmst/agents.git /app/agents \
  && cd /app/agents \
  && git reset --hard ${AGENTS_REV?} \
  && pip --no-cache-dir install -e .

# optional wandb installation
RUN pip --no-cache-dir install wandb
