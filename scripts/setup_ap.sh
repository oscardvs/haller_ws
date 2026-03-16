#!/bin/bash
# Set up Haller Robot WiFi Access Point
# Usage: sudo ./setup_ap.sh [start|stop]
#
# When AP is active:
#   - SSID: HallerRobot
#   - Password: haller2024
#   - Robot IP: 10.42.0.1
#   - Web teleop: http://10.42.0.1:8080

SSID="HallerRobot"
PASSWORD="haller2024"
CON_NAME="HallerAP"
WIFI_DEV="wlP1p1s0"

case "${1:-start}" in
    start)
        echo "[haller] Starting WiFi Access Point..."
        # Delete existing AP connection if any
        nmcli con delete "$CON_NAME" 2>/dev/null
        # Create hotspot
        nmcli con add type wifi ifname "$WIFI_DEV" con-name "$CON_NAME" \
            autoconnect no ssid "$SSID"
        nmcli con modify "$CON_NAME" \
            802-11-wireless.mode ap \
            802-11-wireless.band bg \
            802-11-wireless.channel 6 \
            ipv4.method shared \
            ipv4.addresses 10.42.0.1/24 \
            wifi-sec.key-mgmt wpa-psk \
            wifi-sec.psk "$PASSWORD"
        nmcli con up "$CON_NAME"
        echo "[haller] AP active: SSID=$SSID, IP=10.42.0.1"
        echo "[haller] Web teleop: http://10.42.0.1:8080"
        ;;
    stop)
        echo "[haller] Stopping WiFi Access Point..."
        nmcli con down "$CON_NAME" 2>/dev/null
        nmcli con delete "$CON_NAME" 2>/dev/null
        # Reconnect to previous WiFi
        nmcli device wifi rescan
        nmcli device connect "$WIFI_DEV" 2>/dev/null
        echo "[haller] AP stopped, reconnecting to WiFi..."
        ;;
    *)
        echo "Usage: $0 [start|stop]"
        exit 1
        ;;
esac
