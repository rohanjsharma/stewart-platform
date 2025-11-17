import serial
import json
import time
import numpy as np

from random import randint
class ServoBus:
    def __init__(self):
        """Initialize controller, load config, set defaults and queues."""
        # Load experiment and hardware config from JSON file
        
        
        # Servo port name and center angle
        self.servo_port = "COM6"
        self.neutral_angle = 15.0
        self.servo = None

    def connect_servo(self):
        try:
            self.servo = serial.Serial(self.servo_port, baudrate=9600)
            time.sleep(2)
            print("[SERVO] Connected")
            return True
        except Exception as e:
            print(f"[SERVO] Failed: {e}")
            return False
    
    def send_servo_angle(self, angle1, angle2, angle3):
        """Send angle command to servo motor (clipped for safety)."""
        if self.servo:
            servo_angle1 = self.neutral_angle + angle1
            servo_angle1 = int(np.clip(servo_angle1, 0, 40))
            servo_angle2 = self.neutral_angle + angle2+10.0
            servo_angle2 = int(np.clip(servo_angle2, 0, 40))
            servo_angle3 = self.neutral_angle + angle3 - 4.0
            servo_angle3 = int(np.clip(servo_angle3, 0, 40))
            try:
                self.servo.write(bytes([servo_angle1, servo_angle2, servo_angle3]))
            except Exception:
                print("[SERVO] Send failed")
        
if __name__ == "__main__":
    bus = ServoBus()

    bus.connect_servo()

    
    bus.send_servo_angle(0, 10, -4)  # Example: send 10 degrees offset from neutral
        
