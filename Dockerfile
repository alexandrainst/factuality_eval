FROM astral/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  bash \
  ca-certificates \
  curl \
  git \
  libatomic1 \
  make \
  tar \
  && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 alex-admin \
  && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash alex-admin

# Install mise system-wide
RUN curl https://mise.run | MISE_INSTALL_PATH=/usr/local/bin/mise sh

# Baked-tools pattern: use fixed system paths so shims work for any user
ENV MISE_DATA_DIR="/mise" \
    MISE_CONFIG_DIR="/mise" \
    MISE_CACHE_DIR="/mise/cache" \
    MISE_TRUSTED_CONFIG_PATHS="/workspace" \
    MISE_YES=1

ENV PATH="/mise/shims:${PATH}"

RUN mkdir -p /mise && chown -R alex-admin:alex-admin /mise \
  && mkdir -p /workspace && chown alex-admin:alex-admin /workspace \
  && mkdir -p /workspace/.venv && chown alex-admin:alex-admin /workspace/.venv \
  && rm -rf /home/alex-admin/.cache/uv \
  && mkdir -p /home/alex-admin/.cache/uv \
  && mkdir -p /home/alex-admin/.local/share/uv/python \
  && chown -R alex-admin:alex-admin /home/alex-admin/.cache /home/alex-admin/.local

COPY --chown=alex-admin:alex-admin mise.toml /mise/config.toml

ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/home/alex-admin/.cache/uv
ENV UV_PYTHON_INSTALL_DIR=/home/alex-admin/.local/share/uv/python
ENV PATH="/home/alex-admin/.local/bin:/workspace/.venv/bin:${PATH}"

WORKDIR /workspace

USER alex-admin

RUN mise install

RUN python --version \
  && uv --version \
  && git --version \
  && opencode --version \
  && make --version

ENTRYPOINT ["opencode"]
CMD []
