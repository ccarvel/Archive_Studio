# tkapp/Dockerfile
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
# (Optional) change these if you want different ports
ENV NOVNC_PORT=6080 \
    VNC_PORT=5901 \
    DISPLAY=:1 \
    SCREEN_GEOMETRY=1366x768x24 \
    APP_CMD="python /app/ArchiveStudio.py"

# System deps: Tk, virtual X, minimal WM, VNC, noVNC, websockify, fonts, tini for clean signals
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3-tk tk tcl \
#     xvfb fluxbox x11vnc novnc websockify \
#     xauth x11-xserver-utils \
#     fonts-dejavu-core fonts-dejavu-extra \
#     ca-certificates curl tini \
#  && rm -rf /var/lib/apt/lists/*

# replace the X stack installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk tk tcl \
    tigervnc-standalone-server novnc websockify \
    fluxbox xauth x11-xserver-utils \
    fonts-dejavu-core fonts-dejavu-extra \
    ca-certificates curl tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy your app code into /app (or mount at runtime via volumes)
COPY . /app

# If you have Python deps, uncomment:
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Add start script
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Expose VNC and noVNC ports
EXPOSE ${NOVNC_PORT} ${VNC_PORT}

# Use tini as init for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start.sh"]