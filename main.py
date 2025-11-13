from  PID.pid import PID
from vision.detector2 import BallDetector2D
from vision.circle_mapper import CircleMapper 
from Arduino.ServoBus import ServoBus
import os
import json
import cv2
from math import sqrt, atan2, cos, sin
import numpy as np
from IK.SPV4 import triangle_orientation_and_location
import time
import threading
from PID.pid_tuner_gui import start_pid_tuner

def show_frame_with_overlay(frame, rim_fit, det_result, mapper, ball_x=None, ball_y=None):
    """Draw rim ellipse, ball detection, and position info on frame."""
    vis_frame = frame.copy()
    
    # Draw rim/ellipse if valid
    if rim_fit["valid"]:
        u0, v0 = map(int, rim_fit["center"])
        a, b = int(rim_fit["a"]), int(rim_fit["b"])
        angle_deg = int(rim_fit["psi"] * 180.0 / 3.14159)
        
        # Draw ellipse outline
        cv2.ellipse(vis_frame, (u0, v0), (a, b), angle_deg, 0, 360, (255, 0, 0), 2)
        # Draw center point
        cv2.circle(vis_frame, (u0, v0), 4, (0, 255, 0), -1)
        
        # Status text
        cv2.putText(vis_frame, "Rim OK", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(vis_frame, "Rim not detected", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw ball if detected
    if det_result["found"]:
        u, v = det_result["xy_px"]
        r = int(det_result["radius_px"])
        
        # Draw ball circle
        cv2.circle(vis_frame, (u, v), r, (0, 255, 255), 2)  # Cyan circle
        cv2.circle(vis_frame, (u, v), 3, (0, 255, 255), -1)  # Cyan center dot
        
        # Draw position info if rim is valid and we have meter coordinates
        if rim_fit["valid"] and ball_x is not None and ball_y is not None:
            # Position in meters
            cv2.putText(vis_frame, f"x={ball_x:+.3f}m y={ball_y:+.3f}m", 
                       (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw line from center to ball
            u0, v0 = map(int, rim_fit["center"])
            cv2.line(vis_frame, (u0, v0), (u, v), (0, 255, 255), 1)
        elif rim_fit["valid"]:
            # Convert on the fly if not provided
            x_m, y_m = mapper.px_to_m((u, v))
            cv2.putText(vis_frame, f"x={x_m:+.3f}m y={y_m:+.3f}m", 
                       (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(vis_frame, "Rim not ready", (12, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(vis_frame, "Ball not found", (12, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return vis_frame

def load_cfg():
    # Get path relative to script location, not current working directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    path = os.path.join(PROJECT_ROOT, "config", "config.json")
    
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # Fallback defaults
    return {
        "camera": {"index": 1},
        "vision": {
            "ball_hsv_lower": [10, 120, 120],
            "ball_hsv_upper": [25, 255, 255],
            "erode_iter": 2, "dilate_iter": 2, "morph_kernel": 5,
            "min_radius_px": 8, "max_radius_px": 80, "min_area_frac": 0.0003
        },
        "plate": {"radius_m": 0.15},
        "mapping": { "edge_canny": {"low": 60, "high": 160},
                     "ellipse_refresh_interval": 5 }
    }

servo_bus = ServoBus()

# Connect to servos
if not servo_bus.connect_servo():
    print("[WARNING] Servo connection failed - running in simulation mode")

pid = PID()

# Start PID tuner GUI in separate thread
try:
    tuner_thread = threading.Thread(target=start_pid_tuner, args=(pid,), daemon=True)
    tuner_thread.start()
    time.sleep(0.5)  # Give GUI time to initialize
    print("[INFO] PID Tuner GUI started - adjust gains in real-time!")
except Exception as e:
    print(f"[WARNING] Could not start PID tuner GUI: {e}")
    import traceback
    traceback.print_exc()

cfg = load_cfg()
ball_detector = BallDetector2D(cfg["vision"])
mapper = CircleMapper(cfg.get("plate", {}).get("radius_m", 0.15), cfg.get("mapping", {}))


# SPV4 geometry parameters

platform_z = 14

cap = cv2.VideoCapture(cfg["camera"]["index"])

while True:
    ret, frame = cap.read()
    rim_fit = mapper.update_from_frame(frame)
    det_result = ball_detector.update(frame, return_mask=False)

    ball_x = None
    ball_y = None
    if det_result["found"] and rim_fit["valid"]:
        # C. CONVERT PIXELS TO METERS
        xy_px = det_result["xy_px"]
        ball_x, ball_y = mapper.px_to_m(xy_px)

        print("ball_x", ball_x, "ball_y", ball_y)
        # D. UPDATE PID CONTROLLERS
        phi_x = pid.update_x(ball_x, 0.033)  # Returns degrees
        phi_y = pid.update_y(ball_y, 0.033)  # Returns degrees
        print("phi_x", phi_x, "phi_y", phi_y)

        phi_x = phi_x * np.pi / 180  # Roll angle (rotation around X-axis) in radians
        phi_y = phi_y * np.pi / 180  # Pitch angle (rotation around Y-axis) in radians
        
        # E. CONVERT ROLL/PITCH TO NORMAL VECTOR
        # Roll (phi_x) rotates around X-axis, Pitch (phi_y) rotates around Y-axis
        # Normal vector after rotation: nrm = R_x(roll) * R_y(pitch) * [0, 0, 1]^T
        #n_x = sin(phi_y)
        #n_y = -cos(phi_y) * sin(phi_x)
        #n_z = cos(phi_y) * cos(phi_x)
        #nrm = np.array([n_x, n_y, n_z])
        # Normalize to ensure unit vector
        #nrm = nrm / np.linalg.norm(nrm)
        nrm = np.array([phi_x, phi_y, 1])
        S = np.array([0, 0, 1])
        
        # F. INVERSE KINEMATICS
        ik_result = triangle_orientation_and_location(nrm, S, 0.5)
        
        # G. EXTRACT SERVO ANGLES
        theta_11 = ik_result["theta_11"]  # degrees
        theta_21 = ik_result["theta_21"]  # degrees
        theta_31 = ik_result["theta_31"]  # degrees
        
        # H. CONVERT TO OFFSETS AND SEND
        neutral = 15.0
        offset_11 = 90-(theta_11 - neutral)
        offset_21 = 90-(theta_21 - neutral-10)
        offset_31 = 90-(theta_31 - neutral+4)

        # Check for NaN or invalid values before sending
        if np.isnan(offset_11) or np.isnan(offset_21) or np.isnan(offset_31):
            print("[WARNING] Invalid offset values (NaN) - skipping servo command")
            print(f"  theta_11={theta_11:.2f}°, theta_21={theta_21:.2f}°, theta_31={theta_31:.2f}°")
        else:
            print("offset_11", offset_21, "offset_21", offset_11, "offset_31", offset_31)
            servo_bus.send_servo_angle(offset_11, offset_21, offset_31)

    # I. DISPLAY/UPDATE
    vis_frame = show_frame_with_overlay(frame, rim_fit, det_result, mapper, ball_x, ball_y)
    cv2.imshow("Stewart Platform Control", vis_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    time.sleep(0.033)  # Control loop period

# 4. CLEANUP
servo_bus.send_servo_angle(0, 0, 0)  # Return to neutral
cap.release()