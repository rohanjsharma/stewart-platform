"""
Simple GUI for tuning PID constants in real-time
Run this in a separate thread while main.py is running
"""

import tkinter as tk
from tkinter import ttk
import threading


class PIDTunerGUI:
    def __init__(self, pid_instance):
        """
        Initialize PID tuner GUI.
        
        Args:
            pid_instance: The PID object to tune (from PID.pid import PID)
        """
        self.pid = pid_instance
        self.root = None
        self.running = False
        
    def create_gui(self):
        """Create the GUI window with sliders."""
        self.root = tk.Tk()
        self.root.title("PID Tuner - Stewart Platform")
        self.root.geometry("500x600")
        
        # Title
        title_label = ttk.Label(self.root, text="PID Gain Tuning", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # X-Axis PID Section
        x_frame = ttk.LabelFrame(self.root, text="X-Axis PID", padding=15)
        x_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Kp_X
        ttk.Label(x_frame, text="Kp_X (Proportional)", font=("Arial", 10)).pack()
        self.kp_x_var = tk.DoubleVar(value=self.pid.Kp_x)
        kp_x_slider = ttk.Scale(x_frame, from_=0, to=50, variable=self.kp_x_var,
                               orient=tk.HORIZONTAL, length=400)
        kp_x_slider.pack(pady=5)
        self.kp_x_label = ttk.Label(x_frame, text=f"Kp_X: {self.pid.Kp_x:.2f}")
        self.kp_x_label.pack()
        
        # Ki_X
        ttk.Label(x_frame, text="Ki_X (Integral)", font=("Arial", 10)).pack()
        self.ki_x_var = tk.DoubleVar(value=self.pid.Ki_x)
        ki_x_slider = ttk.Scale(x_frame, from_=0, to=10, variable=self.ki_x_var,
                               orient=tk.HORIZONTAL, length=400)
        ki_x_slider.pack(pady=5)
        self.ki_x_label = ttk.Label(x_frame, text=f"Ki_X: {self.pid.Ki_x:.2f}")
        self.ki_x_label.pack()
        
        # Kd_X
        ttk.Label(x_frame, text="Kd_X (Derivative)", font=("Arial", 10)).pack()
        self.kd_x_var = tk.DoubleVar(value=self.pid.Kd_x)
        kd_x_slider = ttk.Scale(x_frame, from_=0, to=20, variable=self.kd_x_var,
                               orient=tk.HORIZONTAL, length=400)
        kd_x_slider.pack(pady=5)
        self.kd_x_label = ttk.Label(x_frame, text=f"Kd_X: {self.pid.Kd_x:.2f}")
        self.kd_x_label.pack()
        
        # Y-Axis PID Section
        y_frame = ttk.LabelFrame(self.root, text="Y-Axis PID", padding=15)
        y_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Kp_Y
        ttk.Label(y_frame, text="Kp_Y (Proportional)", font=("Arial", 10)).pack()
        self.kp_y_var = tk.DoubleVar(value=self.pid.Kp_y)
        kp_y_slider = ttk.Scale(y_frame, from_=0, to=50, variable=self.kp_y_var,
                               orient=tk.HORIZONTAL, length=400)
        kp_y_slider.pack(pady=5)
        self.kp_y_label = ttk.Label(y_frame, text=f"Kp_Y: {self.pid.Kp_y:.2f}")
        self.kp_y_label.pack()
        
        # Ki_Y
        ttk.Label(y_frame, text="Ki_Y (Integral)", font=("Arial", 10)).pack()
        self.ki_y_var = tk.DoubleVar(value=self.pid.Ki_y)
        ki_y_slider = ttk.Scale(y_frame, from_=0, to=10, variable=self.ki_y_var,
                               orient=tk.HORIZONTAL, length=400)
        ki_y_slider.pack(pady=5)
        self.ki_y_label = ttk.Label(y_frame, text=f"Ki_Y: {self.pid.Ki_y:.2f}")
        self.ki_y_label.pack()
        
        # Kd_Y
        ttk.Label(y_frame, text="Kd_Y (Derivative)", font=("Arial", 10)).pack()
        self.kd_y_var = tk.DoubleVar(value=self.pid.Kd_y)
        kd_y_slider = ttk.Scale(y_frame, from_=0, to=20, variable=self.kd_y_var,
                               orient=tk.HORIZONTAL, length=400)
        kd_y_slider.pack(pady=5)
        self.kd_y_label = ttk.Label(y_frame, text=f"Kd_Y: {self.pid.Kd_y:.2f}")
        self.kd_y_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)
        
        reset_btn = ttk.Button(button_frame, text="Reset Integrals", 
                              command=self.reset_integrals)
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ttk.Button(button_frame, text="Close", 
                              command=self.close_gui)
        close_btn.pack(side=tk.LEFT, padx=5)
        
        # Start update loop
        self.running = True
        self.update_gui()
        
    def update_gui(self):
        """Update PID gains from sliders and refresh display."""
        if self.running and self.root:
            # Update PID gains from sliders
            self.pid.Kp_x = self.kp_x_var.get()
            self.pid.Ki_x = self.ki_x_var.get()
            self.pid.Kd_x = self.kd_x_var.get()
            self.pid.Kp_y = self.kp_y_var.get()
            self.pid.Ki_y = self.ki_y_var.get()
            self.pid.Kd_y = self.kd_y_var.get()
            
            # Update labels
            self.kp_x_label.config(text=f"Kp_X: {self.pid.Kp_x:.2f}")
            self.ki_x_label.config(text=f"Ki_X: {self.pid.Ki_x:.2f}")
            self.kd_x_label.config(text=f"Kd_X: {self.pid.Kd_x:.2f}")
            self.kp_y_label.config(text=f"Kp_Y: {self.pid.Kp_y:.2f}")
            self.ki_y_label.config(text=f"Ki_Y: {self.pid.Ki_y:.2f}")
            self.kd_y_label.config(text=f"Kd_Y: {self.pid.Kd_y:.2f}")
            
            # Schedule next update
            self.root.after(50, self.update_gui)  # Update every 50ms
    
    def reset_integrals(self):
        """Reset PID integral terms to zero."""
        self.pid.integral_x = 0.0
        self.pid.integral_y = 0.0
        print("[PID] Integral terms reset")
    
    def close_gui(self):
        """Close the GUI window."""
        self.running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Start the GUI (call this in a separate thread)."""
        self.create_gui()
        if self.root:
            self.root.mainloop()


def start_pid_tuner(pid_instance):
    """
    Start PID tuner GUI in a separate thread.
    
    Usage in main.py:
        from PID.pid_tuner_gui import start_pid_tuner
        import threading
        
        # After creating pid object:
        tuner_thread = threading.Thread(target=start_pid_tuner, args=(pid,), daemon=True)
        tuner_thread.start()
    """
    tuner = PIDTunerGUI(pid_instance)
    tuner.run()


if __name__ == "__main__":
    # Test the GUI
    from pid import PID
    
    test_pid = PID()
    tuner = PIDTunerGUI(test_pid)
    tuner.run()

