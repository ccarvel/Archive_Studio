#!/usr/bin/env bash
set -euo pipefail

#Archive_Studio/start.sh
# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
: "${DISPLAY:=:1}"
: "${SCREEN_GEOMETRY:=1920x1080x24}"
: "${NOVNC_PORT:=6080}"
: "${VNC_PORT:=5901}"
: "${APP_CMD:=python /app/ArchiveStudio.py}"
: "${KEEP_ALIVE:=true}"   # <--- ADD HERE: enables keep-alive by default

echo "[startup] DISPLAY=${DISPLAY}"
echo "[startup] SCREEN_GEOMETRY=${SCREEN_GEOMETRY}"
echo "[startup] VNC_PORT=${VNC_PORT}, noVNC_PORT=${NOVNC_PORT}"
echo "[startup] APP_CMD=${APP_CMD}"
echo "[startup] KEEP_ALIVE=${KEEP_ALIVE}"

# ---------------------------------------------------------------------
# START SERVICES
# ---------------------------------------------------------------------

# Virtual X server
Xvfb "${DISPLAY}" -screen 0 "${SCREEN_GEOMETRY}" -nolisten tcp &
XVFB_PID=$!
echo "[startup] Xvfb started (pid ${XVFB_PID})"

# Window manager
fluxbox >/tmp/fluxbox.log 2>&1 &
FLUX_PID=$!
echo "[startup] fluxbox started (pid ${FLUX_PID})"

sleep 0.5

# Tkinter app
bash -lc "${APP_CMD}" &
APP_PID=$!
echo "[startup] app started (pid ${APP_PID})"

# VNC server
# x11vnc -display "${DISPLAY}" -rfbport "${VNC_PORT}" -forever -shared -nopw \
#   -repeat -xkb -noxrecord -noxfixes -noxdamage >/tmp/x11vnc.log 2>&1 &
# VNC_PID=$!
# echo "[startup] x11vnc started (pid ${VNC_PID})"

# Start Xvnc (TigerVNC) instead of Xvfb+x11vnc
# Xvnc "${DISPLAY}" -geometry "${SCREEN_GEOMETRY%x*}x${SCREEN_GEOMETRY#*x}" -depth 24 -SecurityTypes None &
# XVNC_PID=$!
x11vnc -display "${DISPLAY}" -rfbport "${VNC_PORT}" -forever -shared -nopw -repeat >/tmp/x11vnc.log 2>&1 &
X11VNC_PID=$!
echo "[startup] x11vnc started (pid ${X11VNC_PID})"

# noVNC
web_dir="/usr/share/novnc"
if [ -d "$web_dir" ]; then
  websockify --web="$web_dir" "${NOVNC_PORT}" "localhost:${VNC_PORT}" >/tmp/novnc.log 2>&1 &
  NOVNC_PID=$!
  echo "[startup] noVNC started (pid ${NOVNC_PID})"
else
  echo "[warn] /usr/share/novnc not found; noVNC UI will be unavailable."
  NOVNC_PID=""
fi

# ---------------------------------------------------------------------
# SHUTDOWN HANDLER
# ---------------------------------------------------------------------
_term() {
  echo "[shutdown] stopping..."
  kill -TERM ${NOVNC_PID:-0} ${X11VNC_PID:-0} ${APP_PID:-0} ${WM_PID:-0} ${XVFB_PID:-0} 2>/dev/null || true
  wait || true
  echo "[shutdown] done."
}
trap _term TERM INT

# ---------------------------------------------------------------------
# KEEP-ALIVE LOGIC
# ---------------------------------------------------------------------
# ⬇️ INSERT THIS SECTION HERE
if [ "$KEEP_ALIVE" = "true" ]; then
  echo "[monitor] Container keep-alive mode enabled."
  # Stay alive forever (for docker -d)
  while true; do sleep 3600; done
else
  # Normal behavior: stop when app exits
  wait ${APP_PID} || true
  _term
fi