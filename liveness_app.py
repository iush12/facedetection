import cv2
import mediapipe as mp
import numpy as np
import time
import random
from enum import Enum

# --- Configuration & Constants ---
class Config:
    # Camera
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    
    # Face Mesh
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7
    
    # Thresholds
    EAR_THRESHOLD = 0.25      # Increased from 0.22 (easier to trigger)
    MAR_THRESHOLD = 0.45      # Mouth Aspect Ratio for smile
    HEAD_YAW_THRESHOLD = 20   # Degrees for head turn
    
    # Challenge Config
    CHALLENGES = {
        "Blink Your Eyes": {"threshold": 0.25, "hold": 0.1},  # Very short hold for blink
        "Smile Broadly":   {"threshold": 0.1, "hold": 0.5},  # Lowered from 0.45
        "Turn Head Left":  {"threshold": 40,   "hold": 0.5},
        "Turn Head Right": {"threshold": 40,   "hold": 0.5}
    }
    
    CHALLENGE_TIME_LIMIT = 10.0  # Increased to give more time
    
    # Colors (BGR)
    COLOR_BG = (30, 30, 30)
    COLOR_TEXT = (255, 255, 255)
    COLOR_ACCENT = (0, 255, 127)    # Spring Green
    COLOR_WARNING = (0, 140, 255)   # Orange
    COLOR_ERROR = (0, 0, 255)       # Red
    COLOR_SUCCESS = (0, 255, 0)     # Green

# --- Enums ---
class AppState(Enum):
    INITIALIZING = 0
    WAITING_FOR_FACE = 1
    ALIGNING_FACE = 2
    CHALLENGE_ACTIVE = 3
    VERIFIED = 4
    FAILED = 5

class ChallengeType(Enum):
    BLINK = "Blink Your Eyes"
    SMILE = "Smile Broadly"
    TURN_LEFT = "Turn Head Left"
    TURN_RIGHT = "Turn Head Right"

# --- Helper Classes ---

class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb_frame)

    @staticmethod
    def get_landmarks(results, w, h):
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None

class GeometryUtils:
    @staticmethod
    def calculate_ear(landmarks, w, h):
        """Calculate Eye Aspect Ratio"""
        # Left eye indices
        left_eye = [33, 160, 158, 133, 153, 144]
        # Right eye indices
        right_eye = [362, 385, 387, 263, 373, 380]
        
        def eye_ratio(indices):
            pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
            # Vertical distances
            v1 = np.linalg.norm(pts[1] - pts[5])
            v2 = np.linalg.norm(pts[2] - pts[4])
            # Horizontal distance
            h_dist = np.linalg.norm(pts[0] - pts[3])
            return (v1 + v2) / (2.0 * h_dist) if h_dist > 0 else 0
        
        return (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2.0

    @staticmethod
    def calculate_mar(landmarks, w, h):
        """Calculate Mouth Aspect Ratio"""
        # Mouth indices
        upper_lip = [13, 81, 311] # Top, left-mid, right-mid
        lower_lip = [14, 178, 402] # Bottom, left-mid, right-mid
        corners = [61, 291] # Left corner, Right corner
        
        pts_upper = np.array([landmarks[13].x * w, landmarks[13].y * h])
        pts_lower = np.array([landmarks[14].x * w, landmarks[14].y * h])
        pts_left = np.array([landmarks[61].x * w, landmarks[61].y * h])
        pts_right = np.array([landmarks[291].x * w, landmarks[291].y * h])
        
        vertical = np.linalg.norm(pts_upper - pts_lower)
        horizontal = np.linalg.norm(pts_left - pts_right)
        
        return vertical / horizontal if horizontal > 0 else 0

    @staticmethod
    def get_head_pose(landmarks, w, h):
        """Estimate head pose (yaw, pitch, roll)"""
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Image points
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h), # Chin
            (landmarks[263].x * w, landmarks[263].y * h), # Left eye left corner
            (landmarks[33].x * w, landmarks[33].y * h),   # Right eye right corner
            (landmarks[291].x * w, landmarks[291].y * h), # Left Mouth corner
            (landmarks[61].x * w, landmarks[61].y * h)    # Right mouth corner
        ], dtype="double")

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        # Calculate Euler angles
        rmat, jac = cv2.Rodrigues(rotation_vector)
        
        # cv2.RQDecomp3x3 returns different number of values depending on version
        # We only need the first value (Euler angles)
        ret = cv2.RQDecomp3x3(rmat)
        angles = ret[0]

        # angles[0] = pitch, angles[1] = yaw, angles[2] = roll
        return angles[0] * 360, angles[1] * 360, angles[2] * 360

# --- Main Application ---

class LivenessApp:
    def __init__(self):
        self.detector = FaceMeshDetector()
        self.state = AppState.INITIALIZING
        self.challenges = []
        self.current_challenge_idx = 0
        self.challenge_start_time = 0
        self.success_start_time = 0 # For holding a pose
        self.message = "Initializing..."
        self.sub_message = ""
        
        # Initialize challenges
        self.reset_challenges()

    def reset_challenges(self):
        # Pick 3 random challenges
        all_challenges = [
            ChallengeType.BLINK,
            ChallengeType.SMILE,
            ChallengeType.TURN_LEFT,
            ChallengeType.TURN_RIGHT
        ]
        self.challenges = random.sample(all_challenges, 3)
        self.current_challenge_idx = 0
        self.state = AppState.WAITING_FOR_FACE

    def check_face_alignment(self, landmarks, w, h):
        """Check if face is centered and at correct distance"""
        nose = landmarks[1]
        
        # Center check
        if nose.x < 0.35 or nose.x > 0.65:
            return False, "Center your face"
        
        # Distance check (using face height)
        top = landmarks[10].y
        bottom = landmarks[152].y
        face_height = bottom - top
        
        if face_height < 0.3:
            return False, "Move Closer"
        if face_height > 0.8:
            return False, "Move Back"
            
        return True, "Perfect"

    def process_challenge(self, challenge, landmarks, w, h):
        """Check if current challenge is met"""
        is_met = False
        cfg = Config.CHALLENGES[challenge.value]
        
        if challenge == ChallengeType.BLINK:
            ear = GeometryUtils.calculate_ear(landmarks, w, h)
            if ear < cfg["threshold"]:
                is_met = True
                
        elif challenge == ChallengeType.SMILE:
            mar = GeometryUtils.calculate_mar(landmarks, w, h)
            if mar > cfg["threshold"]:
                is_met = True
                
        elif challenge == ChallengeType.TURN_LEFT:
            pitch, yaw, roll = GeometryUtils.get_head_pose(landmarks, w, h)
            if yaw < -cfg["threshold"]:
                is_met = True
                
        elif challenge == ChallengeType.TURN_RIGHT:
            pitch, yaw, roll = GeometryUtils.get_head_pose(landmarks, w, h)
            if yaw > cfg["threshold"]:
                is_met = True
                
        return is_met

    def draw_ui(self, frame):
        h, w = frame.shape[:2]
        
        # 1. Background Overlay for Text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), Config.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # 2. Main Message
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.5
        thick = 2
        
        # Determine color based on state
        color = Config.COLOR_TEXT
        if self.state == AppState.VERIFIED: color = Config.COLOR_SUCCESS
        elif self.state == AppState.FAILED: color = Config.COLOR_ERROR
        elif self.state == AppState.CHALLENGE_ACTIVE: color = Config.COLOR_ACCENT
        
        text_size = cv2.getTextSize(self.message, font, scale, thick)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, self.message, (text_x, 60), font, scale, color, thick, cv2.LINE_AA)
        
        # 3. Sub Message
        if self.sub_message:
            scale_sub = 0.8
            text_size_sub = cv2.getTextSize(self.sub_message, font, scale_sub, 1)[0]
            text_x_sub = (w - text_size_sub[0]) // 2
            cv2.putText(frame, self.sub_message, (text_x_sub, 100), font, scale_sub, (200, 200, 200), 1, cv2.LINE_AA)

        # 4. Progress Bar (if in challenge mode)
        if self.state == AppState.CHALLENGE_ACTIVE:
            total_challenges = len(self.challenges)
            bar_w = 300
            bar_h = 10
            bar_x = (w - bar_w) // 2
            bar_y = 110
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            
            # Filled bar
            fill_w = int(bar_w * (self.current_challenge_idx / total_challenges))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), Config.COLOR_ACCENT, -1)
            
            # Timer bar (shrinking)
            time_left = max(0, Config.CHALLENGE_TIME_LIMIT - (time.time() - self.challenge_start_time))
            timer_ratio = time_left / Config.CHALLENGE_TIME_LIMIT
            timer_w = int(w * timer_ratio)
            cv2.rectangle(frame, (0, h-10), (timer_w, h), Config.COLOR_WARNING, -1)

    def draw_debug_info(self, frame, landmarks, w, h):
        """Draw real-time values for debugging"""
        if not landmarks: return
        
        ear = GeometryUtils.calculate_ear(landmarks, w, h)
        mar = GeometryUtils.calculate_mar(landmarks, w, h)
        pitch, yaw, roll = GeometryUtils.get_head_pose(landmarks, w, h)
        
        y_start = 150
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (0, 255, 255) # Yellow
        
        lines = [
            f"DEBUG VALUES:",
            f"EAR: {ear:.3f} (Blink < {Config.EAR_THRESHOLD})",
            f"MAR: {mar:.3f} (Smile > {Config.MAR_THRESHOLD})",
            f"YAW: {yaw:.1f} (Turn > {Config.HEAD_YAW_THRESHOLD})",
            f"PITCH: {pitch:.1f}",
            f"ROLL: {roll:.1f}"
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, y_start + i*line_height), font, scale, color, 1, cv2.LINE_AA)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            results = self.detector.process(frame)
            landmarks = self.detector.get_landmarks(results, w, h)
            
            # State Machine Logic
            if self.state == AppState.WAITING_FOR_FACE:
                self.message = "Looking for Face..."
                self.sub_message = "Please step in front of the camera"
                if landmarks:
                    self.state = AppState.ALIGNING_FACE
            
            elif self.state == AppState.ALIGNING_FACE:
                if landmarks:
                    aligned, msg = self.check_face_alignment(landmarks, w, h)
                    self.message = msg
                    self.sub_message = "Align your face in the center"
                    if aligned:
                        # Start Challenges
                        self.state = AppState.CHALLENGE_ACTIVE
                        self.current_challenge_idx = 0
                        self.challenge_start_time = time.time()
                        self.success_start_time = 0
                else:
                    self.state = AppState.WAITING_FOR_FACE
            
            elif self.state == AppState.CHALLENGE_ACTIVE:
                if not landmarks:
                    self.state = AppState.WAITING_FOR_FACE
                    continue
                
                # Check Timeout
                if time.time() - self.challenge_start_time > Config.CHALLENGE_TIME_LIMIT:
                    self.state = AppState.FAILED
                    self.message = "Time Out!"
                    self.sub_message = "Press 'R' to retry"
                    continue
                
                current_challenge = self.challenges[self.current_challenge_idx]
                challenge_name = current_challenge.value
                challenge_cfg = Config.CHALLENGES[challenge_name]
                
                self.message = challenge_name
                self.sub_message = f"Challenge {self.current_challenge_idx + 1}/{len(self.challenges)}"
                
                # Verify Challenge
                if self.process_challenge(current_challenge, landmarks, w, h):
                    if self.success_start_time == 0:
                        self.success_start_time = time.time()
                    
                    # Calculate progress
                    hold_duration = challenge_cfg["hold"]
                    elapsed = time.time() - self.success_start_time
                    
                    # Visual feedback for holding
                    if hold_duration > 0.1: # Only show for long holds
                        progress = min(1.0, elapsed / hold_duration)
                        cv2.rectangle(frame, (0, h-20), (int(w * progress), h), Config.COLOR_SUCCESS, -1)
                        self.sub_message = "Perfect! Hold it..."
                    
                    # Check if completed
                    if elapsed > hold_duration:
                        self.current_challenge_idx += 1
                        self.success_start_time = 0
                        self.challenge_start_time = time.time() # Reset timer for next challenge
                        
                        if self.current_challenge_idx >= len(self.challenges):
                            self.state = AppState.VERIFIED
                            self.message = "LIVENESS VERIFIED"
                            self.sub_message = "Access Granted"
                else:
                    self.success_start_time = 0 # Reset hold if they lose the pose
            
            elif self.state == AppState.VERIFIED:
                # Stay verified until reset
                pass
                
            elif self.state == AppState.FAILED:
                # Wait for reset
                pass
            
            # Draw UI
            self.draw_ui(frame)
            if landmarks:
                self.draw_debug_info(frame, landmarks, w, h)
            
            # Draw Face Mesh (Optional, for debug or cool effect)
            if landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
            
            cv2.imshow('Active Liveness Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_challenges()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    LivenessApp().run()
