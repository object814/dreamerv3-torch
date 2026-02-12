"""
Interactive imagined rollout with web-based keyboard control.

Run:
    python imagine_interactive.py --configs metaworld --checkpoint path/to/checkpoint.pt

Then open http://localhost:5000 in your browser.

Keyboard controls:
    W/S: +/- X axis (0.01)
    A/D: +/- Y axis (0.01)
    Q/E: +/- Z axis (0.01)
    R/F: +/- gripper (0.1)
    Space: No-op (zero action)
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import pathlib
import gymnasium
import envs.wrappers as wrappers
import tools
import models
import argparse
import ruamel.yaml as yaml
import os
import sys
import base64
import threading
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

# Metaworld setup
import envs.metaworld_wrappers as metaworld_wrappers
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioMultiImageObsWrapper


# ------------------------------------------------------------
# HTML Template
# ------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Imagined Rollout</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            margin: 0;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        .image-container {
            border: 3px solid #00d4ff;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        #rollout-image {
            display: block;
            width: 1200px;
            height: 400px;
            object-fit: contain;
            background: #000;
        }
        .controls {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
            background: #16213e;
            color: #00d4ff;
            border: 2px solid #00d4ff;
            transition: all 0.2s;
        }
        button:hover {
            background: #00d4ff;
            color: #1a1a2e;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .info-panel {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            max-width: 600px;
            width: 100%;
        }
        .info-panel h3 {
            color: #00d4ff;
            margin-top: 0;
        }
        .key-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .key-item {
            background: #1a1a2e;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .key-item kbd {
            display: inline-block;
            background: #00d4ff;
            color: #1a1a2e;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .key-item .desc {
            font-size: 12px;
            color: #aaa;
        }
        .status {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .status-item {
            background: #16213e;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .status-label {
            color: #888;
            font-size: 12px;
        }
        .status-value {
            color: #00d4ff;
            font-size: 18px;
            font-weight: bold;
        }
        .action-display {
            font-family: monospace;
            background: #1a1a2e;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        #focus-hint {
            color: #ff6b6b;
            font-size: 14px;
            margin-top: 10px;
            display: none;
        }
        #focus-hint.show {
            display: block;
        }
        .checkpoint-panel {
            background: #16213e;
            padding: 15px 20px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .checkpoint-panel label {
            color: #00d4ff;
            font-weight: bold;
        }
        .checkpoint-panel select {
            padding: 8px 12px;
            font-size: 14px;
            border-radius: 6px;
            border: 2px solid #00d4ff;
            background: #1a1a2e;
            color: #eee;
            min-width: 300px;
            max-width: 500px;
        }
        .checkpoint-panel button {
            padding: 8px 16px;
            font-size: 14px;
        }
        #checkpoint-status {
            color: #888;
            font-size: 12px;
        }
        #checkpoint-status.loading {
            color: #ffcc00;
        }
        #checkpoint-status.success {
            color: #00ff88;
        }
        #checkpoint-status.error {
            color: #ff6b6b;
        }
    </style>
</head>
<body>
    <h1>🤖 Interactive Imagined Rollout</h1>
    
    <div class="container">
        <div class="checkpoint-panel">
            <label>📁 Checkpoint:</label>
            <select id="checkpoint-select">
                <option value="">-- Select checkpoint --</option>
            </select>
            <button onclick="loadCheckpoint()">📥 Load</button>
            <button onclick="refreshCheckpoints()">🔄 Refresh</button>
            <span id="checkpoint-status"></span>
        </div>

        <div class="status">
            <div class="status-item">
                <div class="status-label">Step</div>
                <div class="status-value" id="step-count">0</div>
            </div>
            <div class="status-item">
                <div class="status-label">Reward</div>
                <div class="status-value" id="reward-value">0.00</div>
            </div>
            <div class="status-item">
                <div class="status-label">Total Reward</div>
                <div class="status-value" id="total-reward">0.00</div>
            </div>
            <div class="status-item">
                <div class="status-label">Status</div>
                <div class="status-value" id="status">Idle</div>
            </div>
        </div>

        <div class="image-container">
            <img id="rollout-image" src="" alt="Waiting for image...">
        </div>
        
        <div id="focus-hint" class="show">⚠️ Click anywhere on the page to enable keyboard controls</div>

        <div class="controls">
            <button id="start-btn" onclick="startRollout()">▶ Start</button>
            <button id="reset-btn" onclick="resetRollout()">↻ Reset</button>
            <button id="step-btn" onclick="stepRollout()" disabled>⏭ Step (Space)</button>
            <button id="random-btn" onclick="randomAction()" disabled>🎲 Random</button>
        </div>

        <div class="action-display">
            Last Action: <span id="last-action">[0, 0, 0, 0]</span>
        </div>

        <div class="info-panel">
            <h3>🎮 Keyboard Controls</h3>
            <p>Use keyboard to control the robot in imagination:</p>
            <div class="key-grid">
                <div class="key-item">
                    <kbd>W</kbd>
                    <div class="desc">+X (forward)</div>
                </div>
                <div class="key-item">
                    <kbd>S</kbd>
                    <div class="desc">-X (backward)</div>
                </div>
                <div class="key-item">
                    <kbd>A</kbd>
                    <div class="desc">-Y (left)</div>
                </div>
                <div class="key-item">
                    <kbd>D</kbd>
                    <div class="desc">+Y (right)</div>
                </div>
                <div class="key-item">
                    <kbd>Q</kbd>
                    <div class="desc">+Z (up)</div>
                </div>
                <div class="key-item">
                    <kbd>E</kbd>
                    <div class="desc">-Z (down)</div>
                </div>
                <div class="key-item">
                    <kbd>R</kbd>
                    <div class="desc">Open gripper</div>
                </div>
                <div class="key-item">
                    <kbd>F</kbd>
                    <div class="desc">Close gripper</div>
                </div>
            </div>
            <p style="margin-top: 15px; color: #888;">Press <kbd style="background: #00d4ff; color: #1a1a2e; padding: 2px 6px; border-radius: 3px;">Space</kbd> for no-op (zero action)</p>
        </div>
    </div>

    <script>
        const socket = io();
        let isRunning = false;

        // Key mappings: key -> [x, y, z, gripper]
        const keyActions = {
            'w': [0.03, 0, 0, 0],
            's': [-0.03, 0, 0, 0],
            'a': [0, -0.03, 0, 0],
            'd': [0, 0.03, 0, 0],
            'q': [0, 0, 0.03, 0],
            'e': [0, 0, -0.03, 0],
            'r': [0, 0, 0, 0.1],
            'f': [0, 0, 0, -0.1],
            ' ': [0, 0, 0, 0]
        };

        // Socket event handlers
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            document.getElementById('status').textContent = 'Disconnected';
        });

        socket.on('frame', function(data) {
            document.getElementById('rollout-image').src = 'data:image/jpeg;base64,' + data.image;
            document.getElementById('step-count').textContent = data.step;
            document.getElementById('reward-value').textContent = data.reward.toFixed(2);
            document.getElementById('total-reward').textContent = data.total_reward.toFixed(2);
        });

        socket.on('status', function(data) {
            document.getElementById('status').textContent = data.status;
            isRunning = data.running;
            updateButtons();
        });

        socket.on('initial_frame', function(data) {
            document.getElementById('rollout-image').src = 'data:image/jpeg;base64,' + data.image;
            document.getElementById('step-count').textContent = '0';
            document.getElementById('reward-value').textContent = '0.00';
            document.getElementById('total-reward').textContent = '0.00';
        });

        // Button functions
        function startRollout() {
            socket.emit('start');
        }

        function resetRollout() {
            socket.emit('reset');
        }

        function stepRollout() {
            socket.emit('action', {action: [0, 0, 0, 0]});
            document.getElementById('last-action').textContent = '[0, 0, 0, 0]';
        }

        function randomAction() {
            socket.emit('random_action');
            document.getElementById('last-action').textContent = '[random]';
        }

        function updateButtons() {
            document.getElementById('start-btn').disabled = isRunning;
            document.getElementById('step-btn').disabled = !isRunning;
            document.getElementById('random-btn').disabled = !isRunning;
        }

        // Keyboard handling
        let hasFocus = false;

        document.addEventListener('click', function() {
            hasFocus = true;
            document.getElementById('focus-hint').classList.remove('show');
        });

        document.addEventListener('keydown', function(e) {
            if (!isRunning) return;
            
            const key = e.key.toLowerCase();
            if (key in keyActions) {
                e.preventDefault();
                const action = keyActions[key];
                socket.emit('action', {action: action});
                document.getElementById('last-action').textContent = 
                    '[' + action.map(v => v.toFixed(2)).join(', ') + ']';
            }
        });

        // Prevent scrolling with space
        window.addEventListener('keydown', function(e) {
            if (e.key === ' ' && e.target === document.body) {
                e.preventDefault();
            }
        });

        // Checkpoint handling
        function refreshCheckpoints() {
            socket.emit('list_checkpoints');
        }

        function loadCheckpoint() {
            const select = document.getElementById('checkpoint-select');
            const checkpoint = select.value;
            if (!checkpoint) {
                alert('Please select a checkpoint first');
                return;
            }
            document.getElementById('checkpoint-status').textContent = 'Loading...';
            document.getElementById('checkpoint-status').className = 'loading';
            socket.emit('load_checkpoint', {checkpoint: checkpoint});
        }

        socket.on('checkpoints_list', function(data) {
            const select = document.getElementById('checkpoint-select');
            select.innerHTML = '<option value="">-- Select checkpoint --</option>';
            data.checkpoints.forEach(function(cp) {
                const option = document.createElement('option');
                option.value = cp.path;
                option.textContent = cp.name;
                if (cp.current) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
            document.getElementById('checkpoint-status').textContent = 
                data.checkpoints.length + ' checkpoints found';
            document.getElementById('checkpoint-status').className = '';
        });

        socket.on('checkpoint_loaded', function(data) {
            document.getElementById('checkpoint-status').textContent = 'Loaded: ' + data.name;
            document.getElementById('checkpoint-status').className = 'success';
            isRunning = false;
            updateButtons();
            document.getElementById('status').textContent = 'Ready';
        });

        socket.on('checkpoint_error', function(data) {
            document.getElementById('checkpoint-status').textContent = 'Error: ' + data.error;
            document.getElementById('checkpoint-status').className = 'error';
        });

        // Auto-load checkpoints on connect
        socket.on('connect', function() {
            console.log('Connected to server');
            document.getElementById('status').textContent = 'Connected';
            refreshCheckpoints();
        });
    </script>
</body>
</html>
"""


# ------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------

def format_multicamera_image(img):
    """
    img: (H, W, 3*num_cameras), float in [0,1]
    returns: (H, W*num_cameras, 3), float
    """
    h, w, c = img.shape
    if c > 3:
        num_cameras = c // 3
        frames = [img[:, :, i*3:(i+1)*3] for i in range(num_cameras)]
        img = np.concatenate(frames, axis=1)
    return img


def draw_info(img, reward, step, total_reward):
    """
    img: (H, W, 3), float in [0,1]
    returns: uint8 BGR image
    """
    img = (img * 255).astype(np.uint8)
    cv2.putText(
        img, f"Step: {step}", (5, 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, f"Reward: {reward:.3f}", (5, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, f"Total: {total_reward:.3f}", (5, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return img


def encode_frame(img):
    """Encode image to base64 JPEG for web transmission."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


# ------------------------------------------------------------
# Interactive Imagination Session
# ------------------------------------------------------------

class ImagineSession:
    def __init__(self, wm, env, config, device="cuda"):
        self.wm = wm
        self.env = env
        self.config = config
        self.device = device
        self.action_dim = config.num_actions
        
        self.state = None
        self.step = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.running = False
        self.initial_obs = None
        
    @torch.no_grad()
    def reset(self):
        """Reset environment and initialize imagination state."""
        obs = self.env.reset()
        self.initial_obs = obs
        
        # Prepare obs for encoder
        obs_prep = {k: np.expand_dims(v, 0) for k, v in obs.items() if "log_" not in k}
        obs_prep = self.wm.preprocess(obs_prep)
        
        # Encode
        embed = self.wm.encoder(obs_prep)
        embed = embed.unsqueeze(1)  # (B=1, T=1, E)
        
        # Initial posterior
        post, _ = self.wm.dynamics.observe(
            embed=embed,
            action=torch.zeros(1, 1, self.action_dim, device=self.device),
            is_first=torch.ones(1, 1, device=self.device),
        )
        
        self.state = {k: v[:, -1] for k, v in post.items()}
        self.step = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.running = True
        
        # Return real observation frame
        real_img = obs["image"].astype(np.float32) / 255.0
        real_img = format_multicamera_image(real_img)
        real_img = draw_info(real_img, 0.0, 0, 0.0)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
        
        return encode_frame(real_img)
    
    @torch.no_grad()
    def imagine_step(self, action):
        """Take one imagination step with the given action."""
        if not self.running or self.state is None:
            return None
        
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        action_tensor = action_tensor.unsqueeze(0)  # (1, action_dim)
        
        # Imagination step
        self.state = self.wm.dynamics.img_step(self.state, action_tensor)
        
        # Get reward
        feat = self.wm.dynamics.get_feat(self.state)
        reward = self.wm.heads["reward"](feat).mode().item()
        
        # Decode image
        recon = self.wm.heads["decoder"](feat.unsqueeze(1))["image"].mode()
        img = recon[0, 0].cpu().numpy()
        
        # Update stats
        self.step += 1
        self.last_reward = reward
        self.total_reward += reward
        
        # Format for display
        img = format_multicamera_image(img)
        img = draw_info(img, reward, self.step, self.total_reward)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return {
            "image": encode_frame(img),
            "step": self.step,
            "reward": reward,
            "total_reward": self.total_reward,
        }
    
    @torch.no_grad()
    def random_step(self):
        """Take one imagination step with a random action."""
        action = (np.random.rand(self.action_dim) * 2 - 1).tolist()
        return self.imagine_step(action)


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

def make_env(config):
    suite, task = config.task.split("_", 1)
    print("Running DreamerV3 Interactive on Metaworld task:", task)

    env = gymnasium.make(
        "Meta-World/MT1",
        env_name=task,
        render_mode="rgb_array",
        max_episode_steps=config.time_limit,
    )

    env = ProprioMultiImageObsWrapper(
        env,
        image_height=config.size[0],
        image_width=config.size[1],
        camera_names=["topview", "front", "gripperPOV"],
    )

    env = metaworld_wrappers.FirstTerminalObs(env)
    env = metaworld_wrappers.RewardTuningWrapperV2(env)
    env = metaworld_wrappers.Gymnasium2Gym(env)

    env = wrappers.NormalizeActions(env)
    env = wrappers.RewardObs(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    return env


# ------------------------------------------------------------
# Flask App
# ------------------------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dreamerv3-interactive'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
session = None
app_state = {
    'wm': None,
    'env': None,
    'config': None,
    'device': None,
    'checkpoint_dir': None,
    'current_checkpoint': None,
}


def load_checkpoint_into_wm(checkpoint_path):
    """Load a checkpoint into the world model."""
    global session, app_state
    
    device = app_state['device']
    wm = app_state['wm']
    env = app_state['env']
    config = app_state['config']
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent_state = checkpoint["agent_state_dict"]
        
        wm_state = {}
        for k, v in agent_state.items():
            if k.startswith("_wm._orig_mod."):
                wm_state[k.replace("_wm._orig_mod.", "")] = v
        
        wm.load_state_dict(wm_state, strict=False)
        app_state['current_checkpoint'] = checkpoint_path
        
        # Recreate session with new model
        session = ImagineSession(wm, env, config, device)
        
        print(f"Loaded checkpoint: {checkpoint_path}")
        return True, os.path.basename(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False, str(e)


def get_available_checkpoints():
    """Scan checkpoint directory for available checkpoints."""
    checkpoint_dir = app_state['checkpoint_dir']
    current = app_state['current_checkpoint']
    
    checkpoints = []
    
    if checkpoint_dir and os.path.isdir(checkpoint_dir):
        # Search for .pt files
        for root, dirs, files in os.walk(checkpoint_dir):
            for f in files:
                if f.endswith('.pt'):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, checkpoint_dir)
                    checkpoints.append({
                        'path': full_path,
                        'name': rel_path,
                        'current': full_path == current
                    })
    
    # Sort by name
    checkpoints.sort(key=lambda x: x['name'])
    return checkpoints


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'status': 'Ready', 'running': False})


@socketio.on('list_checkpoints')
def handle_list_checkpoints():
    checkpoints = get_available_checkpoints()
    emit('checkpoints_list', {'checkpoints': checkpoints})


@socketio.on('load_checkpoint')
def handle_load_checkpoint(data):
    checkpoint_path = data.get('checkpoint', '')
    if not checkpoint_path:
        emit('checkpoint_error', {'error': 'No checkpoint specified'})
        return
    
    if not os.path.exists(checkpoint_path):
        emit('checkpoint_error', {'error': 'Checkpoint file not found'})
        return
    
    success, msg = load_checkpoint_into_wm(checkpoint_path)
    if success:
        emit('checkpoint_loaded', {'name': msg})
    else:
        emit('checkpoint_error', {'error': msg})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('start')
def handle_start():
    global session
    if session is not None:
        frame = session.reset()
        emit('initial_frame', {'image': frame})
        emit('status', {'status': 'Running', 'running': True})


@socketio.on('reset')
def handle_reset():
    global session
    if session is not None:
        frame = session.reset()
        emit('initial_frame', {'image': frame})
        emit('status', {'status': 'Running', 'running': True})


@socketio.on('action')
def handle_action(data):
    global session
    if session is not None and session.running:
        action = data.get('action', [0, 0, 0, 0])
        # Pad action if needed
        while len(action) < session.action_dim:
            action.append(0.0)
        result = session.imagine_step(action[:session.action_dim])
        if result:
            emit('frame', result)


@socketio.on('random_action')
def handle_random_action():
    global session
    if session is not None and session.running:
        result = session.random_step()
        if result:
            emit('frame', result)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(config):
    global session, app_state
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tools.set_seed_everywhere(config.seed)

    env = make_env(config)
    obs_space = env.observation_space
    act_space = env.action_space
    config.num_actions = act_space.shape[0]

    # ---- world model ----
    wm = models.WorldModel(
        obs_space=obs_space,
        act_space=act_space,
        step=0,
        config=config,
    ).to(device)

    wm.requires_grad_(False)
    wm.eval()

    # ---- store global state ----
    app_state['wm'] = wm
    app_state['env'] = env
    app_state['config'] = config
    app_state['device'] = device
    app_state['checkpoint_dir'] = config.checkpoint_dir
    
    # ---- load initial checkpoint ----
    checkpoint = torch.load(config.checkpoint, map_location=device)
    agent_state = checkpoint["agent_state_dict"]

    wm_state = {}
    for k, v in agent_state.items():
        if k.startswith("_wm._orig_mod."):
            wm_state[k.replace("_wm._orig_mod.", "")] = v

    wm.load_state_dict(wm_state, strict=False)
    app_state['current_checkpoint'] = config.checkpoint
    print("Loaded world model.")

    # ---- create session ----
    session = ImagineSession(wm, env, config, device)
    
    # ---- start server ----
    host = config.host
    port = config.port
    print(f"\n{'='*60}")
    print(f"Interactive Imagination Server")
    print(f"{'='*60}")
    print(f"Open in browser: http://{host}:{port}")
    print(f"If on remote server, use SSH tunnel:")
    print(f"  ssh -L {port}:localhost:{port} user@server")
    print(f"Then open: http://localhost:{port}")
    print(f"{'='*60}\n")
    
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


# ------------------------------------------------------------
# Config parsing
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                recursive_update(base[k], v)
            else:
                base[k] = v

    defaults = {}
    for name in ["defaults", *(args.configs or [])]:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for k, v in sorted(defaults.items()):
        parser.add_argument(f"--{k}", type=tools.args_type(v), default=v)

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="logdir", help="Directory to scan for checkpoints")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    
    main(parser.parse_args(remaining))
