import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import csv
import math
from main import distance, open_palm, extract_normalized_features

# Configuration
current_label = None
required_hands = 1  # Default single hand

baseOptions = python.BaseOptions
HandLandMarker = vision.HandLandmarker
HandLandMarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

hand_model_path = 'hand_landmarker.task'
hand_options = HandLandMarkerOptions(
    base_options=baseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2  # Always detect 2 for Hollow Purple Prepare
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
] 

hand_landmarker = HandLandMarker.create_from_options(hand_options)
cap = cv2.VideoCapture(0)
start_time = time.time()

# Tracking
label_counts = {'Red': 0, 'Blue': 0, 'Hollow Purple Prepare': 0, 
                'Hollow Purple Snap': 0, 'Domain Expansion': 0, 'Nothing': 0}
max_samples = 1500
buffer = []
sequence_id = 0
state = 'idle'
target_hand = None
neutral_frame_count = 0

# Separate CSV files for different feature dimensions
single_hand_file = 'gesture_data_single.csv'
dual_hand_file = 'gesture_data_dual.csv'

# Initialize single-hand CSV (42 features: x0-x20, y0-y20)
with open(single_hand_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['seq_id', 'frame_idx'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label']
    writer.writerow(header)

# Initialize dual-hand CSV (85 features: left 42 + right 42 + wrist_distance)
with open(dual_hand_file, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['seq_id', 'frame_idx'] + [f'lx{i}' for i in range(21)] + [f'ly{i}' for i in range(21)] + \
             [f'rx{i}' for i in range(21)] + [f'ry{i}' for i in range(21)] + ['wrist_dist', 'label']
    writer.writerow(header)
    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    key = cv2.waitKey(1) & 0xFF
    
    # Frame processing
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    
    # Phase 1: Extract all hand data
    hands_data = []
    raw_landmarks = []  # Store for dual-hand fusion
    
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            features, max_dist = extract_normalized_features(hand_landmarks)
            
            # Visualization
            points = []
            for lm in hand_landmarks:
                points.append((int(lm.x * w), int(lm.y * h)))
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (132, 32, 197), 1)
            
            for a, b in HAND_CONNECTIONS:
                if a < len(points) and b < len(points):
                    cv2.line(frame, points[a], points[b], (178, 205, 255), 1)
            
            hands_data.append({
                'features': features,
                'max_dist': max_dist,
                'open_palm': open_palm(hand_landmarks),
                'points': points
            })
            raw_landmarks.append(hand_landmarks)
    
    # Phase 2: State Machine
    if state == 'idle':
        if key == ord('c'):
            if current_label is None:
                print('ERROR: Select label first (1,2,3,4,5,n)')
            elif label_counts[current_label] >= max_samples:
                print(f'Label {current_label} already completed!')
            else:
                state = 'waiting_for_neutral'
                neutral_frame_count = 0
                target_hand = None
                print(f'Get Ready for {current_label}')
    
    elif state == 'waiting_for_neutral':
        if required_hands == 2:
            # Dual hand: Hollow Purple Prepare
            if len(hands_data) != 2:
                cv2.putText(frame, 'Show BOTH hands', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                neutral_frame_count = 0
            elif not (hands_data[0]['open_palm'] and hands_data[1]['open_palm']):
                cv2.putText(frame, 'Open BOTH palms', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                neutral_frame_count = 0
            else:
                neutral_frame_count += 1
                cv2.putText(frame, f'Hold... {neutral_frame_count}/10', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if neutral_frame_count >= 10:
                    state = 'recording'
                    buffer = []
                    print('Go! Merge hands together')
        
        else:
            # Single hand: Red, Blue, Snap, Domain, Nothing
            if not hands_data:
                cv2.putText(frame, 'Show hand', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                target_hand = None
                neutral_frame_count = 0
            else:
                # Select first open palm hand, or first hand if none open
                selected = None
                for i, hand in enumerate(hands_data):
                    if hand['open_palm']:
                        selected = i
                        break
                if selected is None:
                    selected = 0
                
                if target_hand is None or target_hand != selected:
                    target_hand = selected
                    neutral_frame_count = 1
                else:
                    neutral_frame_count += 1
                
                cv2.putText(frame, f'Hold... {neutral_frame_count}/10', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if neutral_frame_count >= 10:
                    state = 'recording'
                    buffer = []
                    print('Go!')
    
    elif state == 'recording':
        if required_hands == 2:
            # Dual hand recording: Hollow Purple Prepare
            if len(hands_data) != 2:
                print('Lost hand! Aborting.')
                state = 'idle'
                buffer = []
                continue
            
            left_features = hands_data[0]['features']
            right_features = hands_data[1]['features']
            
            # Calculate wrist distance
            left_wrist = raw_landmarks[0][0]
            right_wrist = raw_landmarks[1][0]
            wrist_dist = distance(left_wrist, right_wrist)
            
            # Normalize by average hand scale
            scale = (hands_data[0]['max_dist'] + hands_data[1]['max_dist']) / 2
            if scale == 0:
                scale = 1
            
            fused_features = left_features + right_features + [wrist_dist / scale]
            
            buffer.append({
                'features': fused_features,
                'timestamp': timestamp_ms
            })
            
            # Progress bar
            progress = len(buffer) / 100
            bar_width = int(w * progress)
            cv2.rectangle(frame, (0, h-20), (bar_width, h), (0, 255, 0), -1)
            cv2.putText(frame, f'MERGE: {len(buffer)}/100', (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if len(buffer) >= 100:
                # Save to dual-hand CSV
                with open(dual_hand_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for frame_idx, frame_data in enumerate(buffer):
                        row = [sequence_id, frame_idx] + frame_data['features'] + [current_label]
                        writer.writerow(row)
                
                label_counts[current_label] += 1
                sequence_id += 1
                print(f'Saved dual sequence {sequence_id}')
                
                if label_counts[current_label] >= max_samples:
                    print(f'{current_label} completed!')
                
                state = 'idle'
                buffer = []
        
        else:
            # Single hand recording
            if target_hand is None or target_hand >= len(hands_data):
                print('Lost hand! Aborting.')
                state = 'idle'
                buffer = []
                continue
            
            hand = hands_data[target_hand]
            buffer.append({
                'features': hand['features'],
                'timestamp': timestamp_ms
            })
            
            # Progress bar
            progress = len(buffer) / 100
            bar_width = int(w * progress)
            cv2.rectangle(frame, (0, h-20), (bar_width, h), (0, 255, 0), -1)
            cv2.putText(frame, f'RECORD: {len(buffer)}/100', (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if len(buffer) >= 100:
                # Save to single-hand CSV
                with open(single_hand_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for frame_idx, frame_data in enumerate(buffer):
                        row = [sequence_id, frame_idx] + frame_data['features'] + [current_label]
                        writer.writerow(row)
                
                label_counts[current_label] += 1
                sequence_id += 1
                print(f'Saved single sequence {sequence_id}')
                
                if label_counts[current_label] >= max_samples:
                    print(f'{current_label} completed!')
                
                state = 'idle'
                buffer = []
                target_hand = None
    
    # Status display
    y_offset = 60
    for label, count in label_counts.items():
        color = (0, 255, 0) if count >= max_samples else (0, 255, 255)
        status = 'DONE' if count >= max_samples else f'{count}/{max_samples}'
        cv2.putText(frame, f'{label[:15]}: {status}', (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 20
    
    cv2.imshow('Data Collection', frame)
    
    # Key handling
    if key == ord('q'):
        break
    elif key == ord('1'):
        current_label = 'Red'
        required_hands = 1
        print('Label: Red (1 hand)')
    elif key == ord('2'):
        current_label = 'Blue'
        required_hands = 1
        print('Label: Blue (1 hand)')
    elif key == ord('3'):
        current_label = 'Hollow Purple Prepare'
        required_hands = 2
        print('Label: Hollow Purple Prepare (2 hands)')
    elif key == ord('4'):
        current_label = 'Hollow Purple Snap'
        required_hands = 1
        print('Label: Hollow Purple Snap (1 hand)')
    elif key == ord('5'):
        current_label = 'Domain Expansion'
        required_hands = 1
        print('Label: Domain Expansion (1 hand)')
    elif key == ord('n'):
        current_label = 'Nothing'
        required_hands = 1
        print('Label: Nothing (1 hand)')
    elif key == ord('s'):
        state = 'idle'
        buffer = []
        target_hand = None
        print('Stopped')

# Cleanup
print('Exiting...')
cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()