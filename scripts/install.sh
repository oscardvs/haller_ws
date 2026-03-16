#!/bin/bash
# Haller Robot - Install autostart services
# Run with: sudo ./scripts/install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Haller Robot Installation ==="

# 1. Install udev rules
echo "[1/4] Installing udev rules..."
cp "$SCRIPT_DIR/99-haller-devices.rules" /etc/udev/rules.d/
udevadm control --reload-rules
udevadm trigger
echo "  -> /dev/haller_lidar and /dev/haller_can symlinks configured"

# 2. Install AP service
echo "[2/4] Installing WiFi AP service..."
cp "$SCRIPT_DIR/haller-ap.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable haller-ap.service
echo "  -> haller-ap.service enabled"

# 3. Install robot service
echo "[3/4] Installing ROS bringup service..."
cp "$SCRIPT_DIR/haller-robot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable haller-robot.service
echo "  -> haller-robot.service enabled"

# 4. Add orin to dialout group (for serial access)
echo "[4/4] Ensuring user permissions..."
usermod -aG dialout orin 2>/dev/null || true

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Services will start on next boot. To start now:"
echo "  sudo systemctl start haller-ap"
echo "  sudo systemctl start haller-robot"
echo ""
echo "To connect:"
echo "  1. Connect to WiFi: HallerRobot (password: haller2024)"
echo "  2. Open browser: http://10.42.0.1:8080"
echo ""
echo "To stop AP and reconnect to regular WiFi:"
echo "  sudo ./scripts/setup_ap.sh stop"
echo ""
echo "To disable autostart:"
echo "  sudo systemctl disable haller-ap haller-robot"
