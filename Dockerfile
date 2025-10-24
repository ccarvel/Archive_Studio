# tkapp/Dockerfile
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV NOVNC_PORT=6080 \
    VNC_PORT=5901 \
    DISPLAY=:1 \
    SCREEN_GEOMETRY=1920x1080x24 \
    APP_CMD="python /app/ArchiveStudio.py"

# System deps: Tk, virtual X, WM, x11vnc, noVNC, websockify, fonts, tini
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk tk tcl \
    xvfb x11vnc fluxbox novnc websockify \
    xauth x11-xserver-utils \
    wmctrl xdotool x11-utils \
    fonts-dejavu-core fonts-dejavu-extra \
    ca-certificates curl tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

EXPOSE ${NOVNC_PORT} ${VNC_PORT}

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/start.sh"]