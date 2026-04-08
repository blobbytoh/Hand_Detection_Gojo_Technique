import math
import csv


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2) + ((p1.y - p2.y)**2)

def open_palm(hand_landmarks):
    wrist = hand_landmarks[0]
    fingers = [(5, 8), (9, 12), (13, 16), (17, 20)]
    extended = 0
    
    for knuckle, tip in fingers:
        knuckle_point = hand_landmarks[knuckle]
        tip_point = hand_landmarks[tip]
        
        if distance(wrist, tip_point) > distance(wrist, knuckle_point):
            extended += 1
            
    return extended >= 4

def extract_normalized_features(hand_landmarks):
    """Extract 42 normalized features from single hand."""
    wrist = hand_landmarks[0]
    translated = []
    
    for lm in hand_landmarks:
        tx = lm.x - wrist.x
        ty = lm.y - wrist.y
        translated.extend([tx, ty])
    
    # Find max distance for normalization
    max_dist = 0.0
    for i in range(0, len(translated), 2):
        dist = math.sqrt(translated[i]**2 + translated[i+1]**2)
        if dist > max_dist:
            max_dist = dist
    
    if max_dist == 0:
        max_dist = 1.0
    
    # Normalize
    features = [v / max_dist for v in translated]
    return features, max_dist


