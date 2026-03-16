#!/bin/bash
# Haller Robot - Install autostart services
# Run with: sudo ./scripts/install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Haller Robot Installation ==="

# 1. Install udev rules
echo "[1/3] Installing udev rules..."
cp "$SCRIPT_DIR/99-haller-devices.rules" /etc/udev/rules.d/
udevadm control --reload-rules
udevadm trigger
echo "  -> /dev/haller_lidar and /dev/haller_can symlinks configured"

# 2. Install robot service
echo "[2/3] Installing ROS bringup service..."
cp "$SCRIPT_DIR/haller-robot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable haller-robot.service
echo "  -> haller-robot.service enabled"

# 3. Add orin to dialout group (for serial access)
echo "[3/3] Ensuring user permissions..."
usermod -aG dialout orin 2>/dev/null || true

# 4. Add ROS sourcing to .bashrc if not already there
if ! grep -q "ros/humble/setup.bash" /home/orin/.bashrc; then
    cat >> /home/orin/.bashrc << 'ROSEOF'

# ROS 2 Humble
source /opt/ros/humble/setup.bash
source /home/orin/haller_ws/install/setup.bash
ROSEOF
    echo "  -> Added ROS sourcing to .bashrc"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Services will start on next boot. To start now:"
echo "  sudo systemctl start haller-robot"
echo ""
echo "Web teleop: http://192.168.0.124:8080"
echo "(from any device on the same WiFi network)"
echo ""
echo "To disable autostart:"
echo "  sudo systemctl disable haller-robot"
