#!/usr/bin/env python3
"""
Web Teleop Node for Haller Robot

Serves a mobile-friendly web interface on port 8080 for driving the robot.
Uses only Python stdlib - no external dependencies needed.

Browse to http://<robot-ip>:8080 to control the robot.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Haller Robot</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, system-ui, sans-serif;
    background: #1a1a2e;
    color: #eee;
    height: 100vh;
    overflow: hidden;
    touch-action: none;
    -webkit-user-select: none;
    user-select: none;
}
.header {
    text-align: center;
    padding: 8px;
    background: #16213e;
    font-size: 14px;
}
.header .status {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    background: #e74c3c;
}
.header .status.ok { background: #2ecc71; }
.main {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: calc(100vh - 40px);
    gap: 16px;
}
.info {
    font-size: 13px;
    color: #888;
    text-align: center;
}
.info span { color: #eee; font-weight: 600; }
.joystick-area {
    position: relative;
    width: 240px;
    height: 240px;
    border-radius: 50%;
    background: radial-gradient(circle, #1e3a5f 0%, #16213e 70%);
    border: 2px solid #334;
    touch-action: none;
}
.joystick-knob {
    position: absolute;
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #4a90d9, #2c5f8a);
    border: 2px solid #5ca0e8;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    transition: none;
    box-shadow: 0 2px 12px rgba(74,144,217,0.4);
}
.speed-control {
    display: flex;
    gap: 12px;
    align-items: center;
}
.speed-control label { font-size: 13px; color: #888; }
.speed-control input[type=range] {
    width: 160px;
    accent-color: #4a90d9;
}
.speed-val { width: 40px; text-align: center; font-size: 13px; }
.btn-stop {
    background: #c0392b;
    color: white;
    border: none;
    padding: 14px 48px;
    font-size: 16px;
    font-weight: 700;
    border-radius: 8px;
    cursor: pointer;
    letter-spacing: 1px;
}
.btn-stop:active { background: #e74c3c; }
.keys-hint {
    font-size: 11px;
    color: #555;
    text-align: center;
}
</style>
</head>
<body>

<div class="header">
    <span class="status" id="statusDot"></span>
    <span id="statusText">Connecting...</span>
</div>

<div class="main">
    <div class="info">
        Linear: <span id="linVel">0.00</span> m/s &nbsp;|&nbsp;
        Angular: <span id="angVel">0.00</span> rad/s
    </div>

    <div class="joystick-area" id="joystickArea">
        <div class="joystick-knob" id="joystickKnob"></div>
    </div>

    <div class="speed-control">
        <label>Max Speed</label>
        <input type="range" id="maxSpeed" min="0.1" max="1.0" step="0.1" value="0.5">
        <span class="speed-val" id="maxSpeedVal">0.5</span>
    </div>

    <button class="btn-stop" id="btnStop">STOP</button>

    <div class="keys-hint">Desktop: WASD or Arrow Keys | Mobile: Touch joystick</div>
</div>

<script>
const API = '';
let linear = 0, angular = 0;
let maxSpeed = 0.5;
let connected = false;
let sendInterval = null;

const $ = id => document.getElementById(id);

// --- Joystick ---
const area = $('joystickArea');
const knob = $('joystickKnob');
let dragging = false;

function getJoystickPos(e) {
    const rect = area.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const r = rect.width / 2 - 35;
    let clientX, clientY;
    if (e.touches) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }
    let dx = clientX - cx;
    let dy = clientY - cy;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist > r) { dx = dx/dist*r; dy = dy/dist*r; }
    return { x: dx/r, y: -dy/r };
}

function updateKnob(nx, ny) {
    const r = area.offsetWidth / 2 - 35;
    knob.style.transform = `translate(calc(-50% + ${nx*r}px), calc(-50% + ${-ny*r}px))`;
}

function onMove(e) {
    if (!dragging) return;
    e.preventDefault();
    const pos = getJoystickPos(e);
    linear = pos.y * maxSpeed;
    angular = -pos.x * maxSpeed * 2;
    updateKnob(pos.x, pos.y);
    updateDisplay();
}

function onEnd() {
    dragging = false;
    linear = 0; angular = 0;
    updateKnob(0, 0);
    updateDisplay();
    sendCmd();
}

area.addEventListener('mousedown', e => { dragging = true; onMove(e); });
area.addEventListener('touchstart', e => { dragging = true; onMove(e); }, {passive: false});
window.addEventListener('mousemove', onMove);
window.addEventListener('touchmove', onMove, {passive: false});
window.addEventListener('mouseup', onEnd);
window.addEventListener('touchend', onEnd);

// --- Keyboard ---
const keys = {};
window.addEventListener('keydown', e => {
    keys[e.key] = true;
    updateFromKeys();
});
window.addEventListener('keyup', e => {
    delete keys[e.key];
    updateFromKeys();
});

function updateFromKeys() {
    let l = 0, a = 0;
    if (keys['w'] || keys['ArrowUp']) l += maxSpeed;
    if (keys['s'] || keys['ArrowDown']) l -= maxSpeed;
    if (keys['a'] || keys['ArrowLeft']) a += maxSpeed * 2;
    if (keys['d'] || keys['ArrowRight']) a -= maxSpeed * 2;
    if (!dragging) {
        linear = l; angular = a;
        updateDisplay();
    }
}

// --- Stop button ---
$('btnStop').addEventListener('click', () => {
    linear = 0; angular = 0;
    updateKnob(0, 0);
    updateDisplay();
    sendCmd();
});

// --- Speed slider ---
$('maxSpeed').addEventListener('input', e => {
    maxSpeed = parseFloat(e.target.value);
    $('maxSpeedVal').textContent = maxSpeed.toFixed(1);
});

// --- Display ---
function updateDisplay() {
    $('linVel').textContent = linear.toFixed(2);
    $('angVel').textContent = angular.toFixed(2);
}

// --- Send commands ---
function sendCmd() {
    fetch(API + '/cmd_vel', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({linear: linear, angular: angular})
    }).then(r => {
        if (!connected) {
            connected = true;
            $('statusDot').classList.add('ok');
            $('statusText').textContent = 'Connected';
        }
        return r.json();
    }).catch(() => {
        if (connected) {
            connected = false;
            $('statusDot').classList.remove('ok');
            $('statusText').textContent = 'Disconnected';
        }
    });
}

// Send at 10 Hz
sendInterval = setInterval(sendCmd, 100);

// Initial
updateDisplay();
sendCmd();
</script>
</body>
</html>"""


class TeleopHandler(BaseHTTPRequestHandler):
    """HTTP handler for web teleop."""

    def log_message(self, format, *args):
        # Suppress default HTTP logs
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/status':
            node = self.server.ros_node
            status = {
                'linear': node.last_linear,
                'angular': node.last_angular,
                'odom_x': node.odom_x,
                'odom_y': node.odom_y,
                'min_range': node.min_range,
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/cmd_vel':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                lin = float(data.get('linear', 0.0))
                ang = float(data.get('angular', 0.0))
                # Clamp values
                lin = max(-1.0, min(1.0, lin))
                ang = max(-2.0, min(2.0, ang))
                self.server.ros_node.publish_cmd_vel(lin, ang)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            except (json.JSONDecodeError, ValueError):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"bad request"}')
        else:
            self.send_response(404)
            self.end_headers()


class WebTeleopNode(Node):
    def __init__(self):
        super().__init__('web_teleop')

        self.declare_parameter('port', 8080)
        self.declare_parameter('max_linear', 1.0)
        self.declare_parameter('max_angular', 2.0)

        self.port = self.get_parameter('port').value
        self.max_linear = self.get_parameter('max_linear').value
        self.max_angular = self.get_parameter('max_angular').value

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers for status
        self.create_subscription(Odometry, 'odom', self._odom_cb, 10)
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)

        # State
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.min_range = float('inf')

        # Start HTTP server in background thread
        self.http_server = HTTPServer(('0.0.0.0', self.port), TeleopHandler)
        self.http_server.ros_node = self
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.http_thread.start()

        self.get_logger().info(f'Web teleop running on http://0.0.0.0:{self.port}')

    def publish_cmd_vel(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = max(-self.max_linear, min(self.max_linear, linear))
        msg.angular.z = max(-self.max_angular, min(self.max_angular, angular))
        self.cmd_pub.publish(msg)
        self.last_linear = msg.linear.x
        self.last_angular = msg.angular.z

    def _odom_cb(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

    def _scan_cb(self, msg: LaserScan):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        self.min_range = min(valid) if valid else float('inf')

    def destroy_node(self):
        self.http_server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
