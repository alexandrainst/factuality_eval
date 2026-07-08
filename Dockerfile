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

ENV MISE_NPM_BUN=false
ENV NPM_CONFIG_IGNORE_SCRIPTS=false

#RUN npm config set ignore-scripts false --location=global

RUN groupadd --gid 1000 user \
  && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash user

# Install mise system-wide
RUN curl https://mise.run | MISE_INSTALL_PATH=/usr/local/bin/mise sh

# Baked-tools pattern: use fixed system paths so shims work for any user
ENV MISE_DATA_DIR="/mise" \
  MISE_CONFIG_DIR="/mise" \
  MISE_CACHE_DIR="/mise/cache" \
  MISE_TRUSTED_CONFIG_PATHS="/workspace" \
  MISE_YES=1

ENV PATH="/mise/shims:${PATH}"

RUN mkdir -p /mise && chown -R user:user /mise \
  && mkdir -p /workspace && chown user:user /workspace \
  && mkdir -p /workspace/.venv && chown user:user /workspace/.venv \
  && rm -rf /home/user/.cache/uv \
  && mkdir -p /home/user/.cache/uv \
  && mkdir -p /home/user/.local/share/uv/python \
  && chown -R user:user /home/user/.cache /home/user/.local

COPY --chown=user:user mise.toml /mise/config.toml

ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/home/user/.cache/uv
ENV UV_PYTHON_INSTALL_DIR=/home/user/.local/share/uv/python
ENV PATH="/home/user/.local/bin:/workspace/.venv/bin:${PATH}"

WORKDIR /workspace

USER user


RUN echo "ignore-scripts=false" > /home/user/.npmrc

RUN mise install -v

# Install caveman for opencode (plugin, commands, skills, AGENTS.md)
RUN npx -y github:JuliusBrussee/caveman -- --only opencode --non-interactive \
  && ls /home/user/.config/opencode/skills/caveman/SKILL.md

RUN python --version \
  && uv --version \
  && git --version \
  && opencode --version \
  && make --version

ENTRYPOINT ["opencode"]
CMD []
