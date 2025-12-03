"""
Professional Face Liveness Detection App

Features:
- Strict Face Visibility Checks (Mask/Hand detection)
- Professional UI with Modern Overlay
- Real-time Liveness Verification
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class LivenessApp:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Parameters
        self.EAR_THRESHOLD = 0.22
        self.BLINK_CONSEC_FRAMES = 2
        self.FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # State
        self.blink_count = 0
        self.blink_counter = 0
        self.is_liveness_verified = False
        self.warning_message = ""
        self.warning_timer = 0
        
        # Colors (BGR)
        self.COLOR_SUCCESS = (0, 255, 100)   # Modern Green
        self.COLOR_WARNING = (0, 140, 255)   # Orange
        self.COLOR_ERROR = (0, 0, 255)       # Red
        self.COLOR_TEXT = (255, 255, 255)    # White
        self.COLOR_BG = (30, 30, 30)         # Dark Gray

    def calculate_ear(self, landmarks, width, height):
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        
        def eye_ratio(indices):
            pts = [np.array([landmarks[i].x * width, landmarks[i].y * height]) for i in indices]
            vertical_1 = np.linalg.norm(pts[1] - pts[5])
            vertical_2 = np.linalg.norm(pts[2] - pts[4])
            horizontal = np.linalg.norm(pts[0] - pts[3])
            return (vertical_1 + vertical_2) / (2.0 * horizontal)
        
        return (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2.0

    def check_obstruction(self, landmarks, width, height):
        """Strict check for face obstruction"""
        warnings = []
        
        # 1. Check Mouth Visibility (Lips shouldn't be covered)
        # Upper lip: 13, Lower lip: 14
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        
        # If lips are too close to nose or chin, might be mask
        nose_tip = landmarks[1]
        chin = landmarks[152]
        
        face_vertical = chin.y - nose_tip.y
        mouth_nose_dist = upper_lip.y - nose_tip.y
        
        # Mask detection logic: if mouth area is flat or textureless (simplified via geometry)
        # Better proxy: Check if mouth landmarks are compressed or erratic
        if mouth_nose_dist < face_vertical * 0.1: 
            warnings.append("Remove Mask / Mouth Covered")

        # 2. Check Eye Visibility
        # If eyes are not detected well, MediaPipe usually fails, but we can check confidence
        # Here we check if eyes are unnaturally closed for too long (handled by blink logic)
        
        # 3. Hand over face check (Depth anomaly)
        # Z-coordinate check: Hands are usually closer than face
        nose_z = nose_tip.z
        # Check cheek points
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        
        # If significant depth variance across face plane
        if abs(left_cheek.z - right_cheek.z) > 0.1:
            warnings.append("Face Obstructed / Remove Hand")

        return warnings

    def check_position(self, landmarks, width, height):
        warnings = []
        forehead = landmarks[10]
        chin = landmarks[152]
        
        # Vertical Position
        face_h = (chin.y - forehead.y) * height
        if face_h < height * 0.35: warnings.append("Move Closer")
        elif face_h > height * 0.85: warnings.append("Move Back")
        
        # Centering
        nose = landmarks[1]
        if nose.x < 0.35 or nose.x > 0.65: warnings.append("Center Your Face")
        
        return warnings

    def draw_modern_ui(self, frame, status_text, status_color, info_text):
        h, w = frame.shape[:2]
        
        # 1. Top Status Bar (Glassmorphism effect)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Status Text with Shadow
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.2
        thick = 2
        text_size = cv2.getTextSize(status_text, font, scale, thick)[0]
        text_x = (w - text_size[0]) // 2
        
        # Shadow
        cv2.putText(frame, status_text, (text_x+2, 52), font, scale, (0,0,0), thick, cv2.LINE_AA)
        # Main Text
        cv2.putText(frame, status_text, (text_x, 50), font, scale, status_color, thick, cv2.LINE_AA)
        
        # 2. Bottom Info Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-60), (w, h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.putText(frame, info_text, (30, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        # 3. Corner Brackets (Viewfinder look)
        color = status_color
        length = 40
        thick = 4
        margin = 30
        
        # Top-Left
        cv2.line(frame, (margin, margin), (margin+length, margin), color, thick)
        cv2.line(frame, (margin, margin), (margin, margin+length), color, thick)
        # Top-Right
        cv2.line(frame, (w-margin, margin), (w-margin-length, margin), color, thick)
        cv2.line(frame, (w-margin, margin), (w-margin, margin+length), color, thick)
        # Bottom-Left
        cv2.line(frame, (margin, h-margin), (margin+length, h-margin), color, thick)
        cv2.line(frame, (margin, h-margin), (margin, h-margin-length), color, thick)
        # Bottom-Right
        cv2.line(frame, (w-margin, h-margin), (w-margin-length, h-margin), color, thick)
        cv2.line(frame, (w-margin, h-margin), (w-margin, h-margin-length), color, thick)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting Professional Liveness Detection...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            status_text = "Align Face"
            status_color = self.COLOR_WARNING
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Checks
                pos_warnings = self.check_position(landmarks, w, h)
                obs_warnings = self.check_obstruction(landmarks, w, h)
                all_warnings = pos_warnings + obs_warnings
                
                if all_warnings:
                    self.warning_message = all_warnings[0]
                    self.warning_timer = time.time()
                    status_text = self.warning_message
                    status_color = self.COLOR_ERROR
                else:
                    if time.time() - self.warning_timer > 1.5:
                        self.warning_message = ""
                        
                        # Liveness Logic
                        ear = self.calculate_ear(landmarks, w, h)
                        if ear < self.EAR_THRESHOLD:
                            self.blink_counter += 1
                        else:
                            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                                self.blink_count += 1
                                self.is_liveness_verified = True
                            self.blink_counter = 0
                        
                        if self.is_liveness_verified:
                            status_text = "VERIFIED"
                            status_color = self.COLOR_SUCCESS
                        else:
                            status_text = "Blink Eyes"
                            status_color = self.COLOR_WARNING
                
                # Draw Oval Guide
                oval_pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]).astype(int) for i in self.FACE_OVAL_INDICES]
                cv2.polylines(frame, [np.array(oval_pts)], True, status_color, 2, cv2.LINE_AA)
                
            else:
                status_text = "No Face Detected"
                status_color = self.COLOR_ERROR
            
            # Draw UI
            info = f"Blinks: {self.blink_count} | Verified: {'YES' if self.is_liveness_verified else 'NO'}"
            self.draw_modern_ui(frame, status_text, status_color, info)
            
            cv2.imshow('Liveness Check', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    LivenessApp().run()
