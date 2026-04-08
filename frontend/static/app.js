import {
    HandLandmarker,
    FilesetResolver,
    FaceLandmarker,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const videoElement = document.getElementById('webcam')
const canvas = document.getElementById('overlay')
const ctx = canvas.getContext('2d')

let handLandmarker = undefined;
let faceLandmarker = undefined
let runningMode = 'VIDEO'

async function createLandmarker(){
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: 'GPU'
        },
        runningMode: runningMode,
        numHands: 2
    })

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'CPU'
        },
        runningMode: runningMode,
        numFaces: 1
    })
}

const ws = new WebSocket(`ws://${window.location.host}/ws`)
ws.onopen = () => {
    console.log('web socket connected')
}

let currentGesture = null;

ws.onmessage = (event) => {
    const result = JSON.parse(event.data)
    currentGesture = result.sign;
    
    // Update the UI display
    const display = document.getElementById('gesture-display');
    if (display && currentGesture) {
        let displayText = currentGesture;
        let borderColor = '#FFEEE2';
        
        if (currentGesture === 'Blue' || currentGesture === 'Red') {
            borderColor = currentGesture === 'Red' ? '#FF4444' : '#44AAFF';

        } else if (currentGesture === 'Hollow Purple Prepare') {
            displayText = 'Red + Blue';
            borderColor = '#AA44FF';

        } else if (currentGesture === 'Hollow Purple Snap') {
            displayText = 'Kyoshiki Murasaki';
            borderColor = '#AA44FF';

        } else if (currentGesture === 'Domain Expansion') {
            displayText = 'Infinite Void';
            borderColor = '#00a2ff';
        }
        
        display.textContent = displayText;
        display.style.borderColor = borderColor;
        display.style.color = borderColor;
    }
}

let lastVideoTime = -1;

async function predictWebcam() {
    if (lastVideoTime !== videoElement.currentTime) {
        lastVideoTime = videoElement.currentTime;

        
        const startTimeMs = performance.now();
        const hand_results = handLandmarker.detectForVideo(videoElement, startTimeMs);
        const face_results = faceLandmarker.detectForVideo(videoElement, startTimeMs);
        
        updateHandLandmarks(hand_results.landmarks)
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(0, 0, 0, 0)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        if (face_results.faceLandmarks){
            for (const landmarks of face_results.faceLandmarks){
                drawFace(landmarks)
            }
        }

        if (hand_results.landmarks) {
            // Draw landmarks
            for (const landmarks of hand_results.landmarks) {
                drawHand(landmarks);
            }
            
            // Send to Python
            if (ws.readyState === WebSocket.OPEN) {
                const handsData = hand_results.landmarks.map(hand => {
                    return hand.map(lm => ({x: lm.x, y: lm.y}));
                });
                ws.send(JSON.stringify({type: 'landmarks', data: handsData}));
            }
        }
    }
    
    requestAnimationFrame(predictWebcam);
}

// Draw hand with connections
function drawHand(landmarks) {
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ];
    
    // Draw connections
    ctx.strokeStyle = '#FFEEE2';  //#FFEEE2
    ctx.lineWidth = 0.8;
    
    for (const [start, end] of connections) {
        const p1 = landmarks[start];
        const p2 = landmarks[end];
        
        ctx.beginPath();
        ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
        ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
        ctx.stroke();
    }
    
    // Draw landmarks
    for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        
        // Color by finger
        let color = '#FFEEE2';
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 1
        ctx.beginPath();
        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
        ctx.stroke()
    }
}

function drawFace(landmarks){
    ctx.fillStyle = '#FFEEE2'

    landmarks.forEach(lm => {
        ctx.beginPath();
        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 1, 0, 2 * Math.PI)
        ctx.fill()
    })
}

function syncCanvasToVideo() {
    const videoAspect = videoElement.videoWidth / videoElement.videoHeight;
    const rect = videoElement.getBoundingClientRect();
    const boxAspect = rect.width / rect.height;

    let renderWidth, renderHeight;

    if (videoAspect > boxAspect) {
        // Pillarboxed — constrained by width
        renderWidth = rect.width;
        renderHeight = rect.width / videoAspect;
    } else {
        // Letterboxed — constrained by height
        renderHeight = rect.height;
        renderWidth = rect.height * videoAspect;
    }

    const offsetX = rect.left + (rect.width - renderWidth) / 2;
    const offsetY = rect.top + (rect.height - renderHeight) / 2;

    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    canvas.style.width = renderWidth + 'px';
    canvas.style.height = renderHeight + 'px';
    canvas.style.top = offsetY + 'px';
}

// Start webcam
async function startWebcam() {
    await createLandmarker();
    
    const constraints = { video: true };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = stream;
    
    videoElement.addEventListener('loadeddata', () => {
        syncCanvasToVideo()
        predictWebcam()
    })
}

window.addEventListener('resize', syncCanvasToVideo)

startWebcam();


// Particles
const mouse = new THREE.Vector2();
window.addEventListener('mousemove', (event) => {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1
})

const particle_canvas = document.getElementById('particles')
const scene = new THREE.Scene()

const particlesGeometry = new THREE.BufferGeometry()
const count = 8000

const posArray = new Float32Array(count * 3);
const initialPositions = new Float32Array(count * 3);
const randomOffset = new Float32Array(count);
const sphereFormationTargets = new Float32Array(count * 3);
const color = new Float32Array(count * 3)

for (let i = 0; i < count; i ++){
    const x = (Math.random() - 0.5) * 100
    const y = (Math.random() - 0.5) * 100
    const z = (Math.random() - 0.5) * 200

    posArray.set([x, y, z], i * 3)
    initialPositions.set([x, y, z], i * 3)
    randomOffset[i] = Math.random() * Math.PI * 2
}

const radius = 4;
const goldenAngle = Math.PI * (3 - Math.sqrt(5));

for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2;
    const radiusAtY = Math.sqrt(1 - y * y);
    const theta = goldenAngle * i;
    
    const x = Math.cos(theta) * radiusAtY;
    const z = Math.sin(theta) * radiusAtY;
    
    sphereFormationTargets[i * 3] = x * radius;
    sphereFormationTargets[i * 3 + 1] = y * radius;
    sphereFormationTargets[i * 3 + 2] = z * radius;
}

// Store original sphere positions for reference
const sphereBasePositions = new Float32Array(sphereFormationTargets);

particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
particlesGeometry.setAttribute('color', new THREE.BufferAttribute(color, 3))

const clock = new THREE.Timer();

const pointsMaterial = new THREE.PointsMaterial({
    size: 0.35,
    color: 0xffffff,
    transparent: true,
    sizeAttenuation: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending,
    vertexColors: true
})

const points = new THREE.Points(particlesGeometry, pointsMaterial)
scene.add(points);

const camera = new THREE.PerspectiveCamera(
    75, 
    window.innerWidth / window.innerHeight,
    0.1,
    100
)
camera.position.z = 50
scene.add(camera)
const renderer = new THREE.WebGLRenderer({
    canvas: particle_canvas,
    antialias: true
})

renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// Smooth mouse tracking with lerp
const targetMouse = new THREE.Vector2();
const currentMouse = new THREE.Vector2();

// Hand landmarks tracking
let currentHandLandmarks = []; // Stores smoothed hand landmark positions
const handSmoothing = 0.2; // Lerp factor for hand movement smoothing

window.addEventListener('mousemove', (event) => {
    targetMouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    targetMouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
});

// Handle resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function updateHandLandmarks(handLandmarksArray){
    if (!handLandmarksArray || handLandmarksArray.length === 0){
        currentHandLandmarks = [];
        return
    }

    currentHandLandmarks = handLandmarksArray.map((hand, handIndex) => {
        return hand.map((lm, lmIndex) => {
            const worldX = (lm.x - 0.5) * 100;
            const worldY = -(lm.y - 0.5) * 100;
            const worldZ = (lm.z || 0) * 50

            if (currentHandLandmarks[handIndex] && currentHandLandmarks[handIndex][lmIndex]){
                const prev = currentHandLandmarks[handIndex][lmIndex];
                return {
                    x: prev.x + (worldX - prev.x) * handSmoothing,
                    y: prev.y + (worldY - prev.y) * handSmoothing,
                    z: prev.z + (worldZ - prev.z) * handSmoothing
                };
            }

            return { x: worldX, y: worldY, z: worldZ }
        });
    });
}

window.updateHandLandmarks = updateHandLandmarks;

let formationProgress = 0
let mergeProgress = 0;
let redFormationProgress = 0;
let blueFormationProgress = 0;
let phaseTimer = 0;
let hollowPhase = 'idle'

let snapProgress = 0;
let snapPhase = 'idle';
let snapTimer = 0;
let snapOrigin = { x: 0, y: 0, z: 0};
let snapCurrentPosition = { x: 0, y: 0, z: 0}
let isMergeComplete = false;

const particleGroup = new Int8Array(count)

for (let i=0; i < count; i ++){
    if (i % 2 == 0){
        particleGroup[i] = 1
    } else{
        particleGroup[i] = 2
    }
}

function animate() {
    requestAnimationFrame(animate);

    clock.update()

    const elapsedTime = clock.getElapsed();
    const currentPositions = particlesGeometry.attributes.position.array;
    const timeScale = elapsedTime * 2.0; // Fast energy

    // Formation progress
    if (currentGesture === 'Red' || currentGesture === 'Blue' || currentGesture === 'Hollow Purple Prepare'){
        formationProgress += 0.045;
    } else if (isMergeComplete){
        formationProgress = 1;
    } else {
        formationProgress -= 0.045;
    }
    formationProgress = Math.max(0, Math.min(1, formationProgress));

    if (currentGesture === 'Hollow Purple Prepare'){
        phaseTimer += 0.016;

        if (hollowPhase == 'idle'){
            hollowPhase = 'red_forming';
        }

        if (hollowPhase === 'red_forming'){
            redFormationProgress += 0.045;

            if (redFormationProgress >= 1){
                redFormationProgress = 1;

                if (phaseTimer > 0.5){
                    hollowPhase = 'blue_forming'
                }
            }
        }

        if (hollowPhase === 'blue_forming'){
            blueFormationProgress += 0.045;

            if (blueFormationProgress >= 1){
                blueFormationProgress = 1;
                if (phaseTimer > 1){
                    hollowPhase = 'merging'
                }
            }
        }

        if (hollowPhase === 'merging'){
            mergeProgress += 0.015
            if (mergeProgress >= 1){
                isMergeComplete = true
            }
        }
    } else {
        redFormationProgress -= 0.045;
        blueFormationProgress -= 0.045;
        mergeProgress -= 0.015;
        phaseTimer = 0;
        hollowPhase = 'idle';
    }

    
    if (!isMergeComplete){
        snapProgress -= 0.02
        if (snapProgress <= 0){
            snapPhase = 'idle'
            snapTimer = 0
        }
    }

    if (isMergeComplete && currentGesture === null || currentHandLandmarks.length === 0) {
        isMergeComplete = false;
        formationProgress -= 0.05
    }

    snapProgress = Math.max(0, Math.min(1, snapProgress));
    redFormationProgress = Math.max(0, Math.min(1, redFormationProgress));
    blueFormationProgress = Math.max(0, Math.min(1, blueFormationProgress));
    mergeProgress = Math.max(0, Math.min(1, mergeProgress));

    currentMouse.x += (targetMouse.x - currentMouse.x) * 0.05;
    currentMouse.y += (targetMouse.y - currentMouse.y) * 0.05;

    // Calculate sphere center ABOVE the hand (index + middle fingertips)
    let centerX = 0, centerY = 0, centerZ = 0;
    let hasHand = false;

    centerX = currentMouse.x * 20;
    centerY = currentMouse.y * 20;
    centerZ = 0;

    if (currentHandLandmarks.length > 0) {
        const hand = currentHandLandmarks[0];

        if (currentGesture === 'Red'){
            const index = hand[8];
            const middle = hand[12];
            
            // Midpoint between index and middle fingertips
            const midX = (index.x + middle.x) / 2;
            const midY = (index.y + middle.y) / 2;
            const midZ = (index.z + middle.z) / 2;
            
            // Position sphere ABOVE the hand (negative Y is up in screen coords)
            centerX = midX;
            centerY = midY + 10; // 15 units ABOVE the fingertips
            centerZ = midZ;
            hasHand = true;
        }else if (currentGesture === "Blue"){
            const index = hand[8]

            centerX = index.x
            centerY = index.y + 10
            centerZ = index.z
            hasHand = true;
        } else if (isMergeComplete){
            const palm = hand[9]
            
            centerX = palm.x
            centerY = palm.y + 10
            centerZ = palm.z || 0;

            hasHand = true
        }

    } else if (currentHandLandmarks.length === 0) {
        isMergeComplete = false;
        formationProgress -= 0.05
    } else {
        centerX = currentMouse.x * 20;
        centerY = currentMouse.y * 20;
        centerZ = 0;
    }
    
    
    if (isMergeComplete && currentGesture === 'Hollow Purple Snap'){
        if (snapPhase === 'idle'){
            snapPhase = 'launching'
            snapOrigin.x = centerX
            snapOrigin.y = centerY
            snapOrigin.z = centerZ

            snapCurrentPosition.x = centerX
            snapCurrentPosition.y = centerY
            snapCurrentPosition.z = centerZ
        }
    }

    if (snapPhase === 'launching'){
        snapProgress += 0.025;

        snapCurrentPosition.x = snapOrigin.x * (1-snapProgress)
        snapCurrentPosition.y = snapOrigin.y * (1-snapProgress)
        snapCurrentPosition.z = snapOrigin.z + (camera.position.z + 5 - snapOrigin.z) * snapProgress

        centerX = snapCurrentPosition.x;
        centerY = snapCurrentPosition.y;
        centerZ = snapCurrentPosition.z;

        if (snapProgress >= 1) {
            // Reset everything after impact
            snapPhase = 'idle';
            snapProgress = 0;
            isMergeComplete = false;
            formationProgress = 0;
            mergeProgress = 0;
            redFormationProgress = 0;
            blueFormationProgress = 0;
            hollowPhase = 'idle';
        }
    }


    // THIS IS THE SPHERE LOGIC
    for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        const offset = randomOffset[i];

        // Floating position (when not formed)
        const floatX = initialPositions[i3] + Math.sin(elapsedTime * 0.2 + offset) * 2;
        const floatY = initialPositions[i3 + 1] + Math.cos(elapsedTime * 0.3 + offset) * 2;
        const floatZ = initialPositions[i3 + 2] + Math.sin(elapsedTime * 0.1 + offset) * 1;

        let sphereX = floatX;
        let sphereY = floatY;
        let sphereZ = floatZ;
        
        if (currentGesture == 'Red' || currentGesture == 'Blue'){
            // Get base sphere position
            const baseX = sphereBasePositions[i3];
            const baseY = sphereBasePositions[i3 + 1];
            const baseZ = sphereBasePositions[i3 + 2];
            
            // 1. Multiple overlapping rotations for spiral effect
            const swirl1 = timeScale * 2.0 + offset;
            
            const cos1 = Math.cos(swirl1 * 0.1);
            const sin1 = Math.sin(swirl1 * 0.1);
            
            // 2. Chaotic radius - never settles
            const breathe = 1 + Math.sin(timeScale * 0.5 + offset) * 0.15;
            const jitter = Math.sin(timeScale * 4 + offset * 5) * 0.4;
            const spike = Math.pow(Math.abs(Math.sin(timeScale * 0.3 + offset)), 4) * 0.4;
            
            const currentRadius = radius * (breathe + jitter) + spike;
            
            // 3. 3D Noise turbulence
            const noiseScale = 1;
            const turbX = Math.sin(baseY * noiseScale + timeScale) * Math.cos(baseZ * noiseScale + timeScale * 0.7) * 0.4;
            const turbY = Math.cos(baseX * noiseScale + timeScale * 0.8) * Math.sin(baseZ * noiseScale + timeScale) * 0.4;
            const turbZ = Math.sin(baseX * noiseScale + timeScale * 0.5) * Math.cos(baseY * noiseScale + timeScale * 0.3) * 0.4;
    
            // 4. Individual particle chaos (each moves independently)
            const chaosSpeed = 10;
            const chaosX = Math.sin(timeScale * chaosSpeed + offset * 15) * 0.5;
            const chaosY = Math.cos(timeScale * chaosSpeed * 0.8 + offset * 15) * 0.5;
            const chaosZ = Math.sin(timeScale * chaosSpeed * 0.6 + offset * 15) * 0.5;
            
            // Apply swirl
            const swirledX = baseX * cos1 - baseZ * sin1;
            const swirledZ = baseX * sin1 + baseZ * cos1;
            
            // Normalize and scale to current radius
            const dist = Math.sqrt(swirledX*swirledX + baseY*baseY + swirledZ*swirledZ) || 1;
            
            // Final sphere position with all chaos applied
            sphereX = (swirledX / dist) * currentRadius + turbX + chaosX + centerX;
            sphereY = (baseY / dist) * currentRadius + turbY + chaosY + centerY;
            sphereZ = (swirledZ / dist) * currentRadius + turbZ + chaosZ + centerZ;

        } else if (currentGesture === 'Hollow Purple Prepare'){
            // Determine which sphere this particle belongs to (permanent assignment)
            const isRedParticle = i % 2 === 0; // Even particles = red, odd = blue
            
            const sphereLocalX = sphereFormationTargets[i3];
            const sphereLocalY = sphereFormationTargets[i3 + 1];
            const sphereLocalZ = sphereFormationTargets[i3 + 2];
            
            // Base positions: Red on right, Blue on left
            let targetCenterX = isRedParticle ? 25 : -25;
            let targetCenterY = 0;
            let targetCenterZ = 0;
            let sphereScale = 1;
            
            // During merge, both move to center
            if (hollowPhase === 'merging') {
                targetCenterX = targetCenterX * (1 - mergeProgress); // Lerp to 0
                sphereScale = 1 + (mergeProgress * 1.5)
            }
            
            // Only animate if this particle's sphere is forming or formed
            const particleProgress = isRedParticle ? redFormationProgress : blueFormationProgress;
            
            if (particleProgress > 0) {
                // Same swirl/turbulence/chaos math as Red/Blue spheres
                const swirl1 = timeScale * 2.0 + offset;
                const cos1 = Math.cos(swirl1 * 0.1);
                const sin1 = Math.sin(swirl1 * 0.1);
                
                const breathe = 1 + Math.sin(timeScale * 0.5 + offset) * 0.15;
                const jitter = Math.sin(timeScale * 4 + offset * 5) * 0.04;
                const spike = Math.pow(Math.abs(Math.sin(timeScale * 0.3 + offset)), 4) * 0.4;
                
                let currentRadius = 7 * (breathe + jitter) + spike;
                currentRadius = currentRadius * sphereScale
                
                const noiseScale = 1;
                const turbX = Math.sin(sphereLocalY * noiseScale + timeScale) * Math.cos(sphereLocalZ * noiseScale + timeScale * 0.7) * 0.2;
                const turbY = Math.cos(sphereLocalX * noiseScale + timeScale * 0.8) * Math.sin(sphereLocalZ * noiseScale + timeScale) * 0.2;
                const turbZ = Math.sin(sphereLocalX * noiseScale + timeScale * 0.5) * Math.cos(sphereLocalY * noiseScale + timeScale * 0.3) * 0.2;
                
                const chaosSpeed = 10;
                const chaosX = Math.sin(timeScale * chaosSpeed + offset * 15) * 0.5;
                const chaosY = Math.cos(timeScale * chaosSpeed * 0.8 + offset * 15) * 0.5;
                const chaosZ = Math.sin(timeScale * chaosSpeed * 0.6 + offset * 15) * 0.5;
                
                const swirledX = sphereLocalX * cos1 - sphereLocalZ * sin1;
                const swirledZ = sphereLocalX * sin1 + sphereLocalZ * cos1;
                
                const dist = Math.sqrt(swirledX*swirledX + sphereLocalY*sphereLocalY + swirledZ*swirledZ) || 1;
                
                sphereX = (swirledX / dist) * currentRadius + turbX + chaosX + targetCenterX;
                sphereY = (sphereLocalY / dist) * currentRadius + turbY + chaosY + targetCenterY;
                sphereZ = (swirledZ / dist) * currentRadius + turbZ + chaosZ + targetCenterZ;
                
                // Blend from float position to sphere position
                sphereX = floatX + (sphereX - floatX) * particleProgress;
                sphereY = floatY + (sphereY - floatY) * particleProgress;
                sphereZ = floatZ + (sphereZ - floatZ) * particleProgress;
            } else {
                // Particle hasn't started forming yet, stay floating
                sphereX = floatX;
                sphereY = floatY;
                sphereZ = floatZ;
            }
        } else if (isMergeComplete){
            const isRedParticle = i % 2 === 0;
    
            const sphereLocalX = sphereFormationTargets[i3];
            const sphereLocalY = sphereFormationTargets[i3 + 1];
            const sphereLocalZ = sphereFormationTargets[i3 + 2];
            
            let targetCenterX = centerX;
            let targetCenterY = centerY;
            let targetCenterZ = centerZ;
            let sphereScale = 2.5;
            
            const swirl1 = timeScale * 2.0 + offset;
            const cos1 = Math.cos(swirl1 * 0.1);
            const sin1 = Math.sin(swirl1 * 0.1);
            
            const breathe = 1 + Math.sin(timeScale * 0.5 + offset) * 0.05;
            const jitter = Math.sin(timeScale * 4 + offset * 5) * 0.15;
            const spike = Math.pow(Math.abs(Math.sin(timeScale * 0.3 + offset)), 4) * 0.2;
            
            let currentRadius = 7 * (breathe + jitter) + spike;
            currentRadius = currentRadius * sphereScale;
            
            const noiseScale = 1;
            const turbX = Math.sin(sphereLocalY * noiseScale + timeScale) * Math.cos(sphereLocalZ * noiseScale + timeScale * 0.7) * 0.15;
            const turbY = Math.cos(sphereLocalX * noiseScale + timeScale * 0.8) * Math.sin(sphereLocalZ * noiseScale + timeScale) * 0.15;
            const turbZ = Math.sin(sphereLocalX * noiseScale + timeScale * 0.5) * Math.cos(sphereLocalY * noiseScale + timeScale * 0.3) * 0.15;
            
            const chaosSpeed = 10;
            const chaosX = Math.sin(timeScale * chaosSpeed + offset * 15) * 0.05;
            const chaosY = Math.cos(timeScale * chaosSpeed * 0.8 + offset * 15) * 0.05;
            const chaosZ = Math.sin(timeScale * chaosSpeed * 0.6 + offset * 15) * 0.05;
            
            const swirledX = sphereLocalX * cos1 - sphereLocalZ * sin1;
            const swirledZ = sphereLocalX * sin1 + sphereLocalZ * cos1;
            
            const dist = Math.sqrt(swirledX*swirledX + sphereLocalY*sphereLocalY + swirledZ*swirledZ) || 1;
            
            sphereX = (swirledX / dist) * currentRadius + turbX + chaosX + targetCenterX;
            sphereY = (sphereLocalY / dist) * currentRadius + turbY + chaosY + targetCenterY;
            sphereZ = (swirledZ / dist) * currentRadius + turbZ + chaosZ + targetCenterZ;
            
            sphereX = floatX + (sphereX - floatX) * 1;
            sphereY = floatY + (sphereY - floatY) * 1;
            sphereZ = floatZ + (sphereZ - floatZ) * 1;
        }

        // Check if we're dispersing (gesture ended)
        const isDispersing = !isMergeComplete && currentGesture === null

        // When dispersing, sphere positions return to float positions
        if (isDispersing) {
            sphereX = floatX;
            sphereY = floatY;
            sphereZ = floatZ;
        }
            
        // Blend
        const targetX = floatX + (sphereX - floatX) * formationProgress;
        const targetY = floatY + (sphereY - floatY) * formationProgress;
        const targetZ = floatZ + (sphereZ - floatZ) * formationProgress;
        
        // Quick movement for energy feel
        const moveSpeed = isDispersing? 0.1 : 0.75;
        currentPositions[i3] += (targetX - currentPositions[i3]) * moveSpeed;
        currentPositions[i3 + 1] += (targetY - currentPositions[i3 + 1]) * moveSpeed;
        currentPositions[i3 + 2] += (targetZ - currentPositions[i3 + 2]) * moveSpeed;
    }

    if (formationProgress > 0 && currentGesture === 'Red') {
        
        const flicker = Math.sin(timeScale * 8) * 0.3 + Math.cos(timeScale * 13) * 0.2;
        const color = particlesGeometry.attributes.color.array;
        
        for (let i = 0; i < count; i ++){
            const i3 = i * 3

            // Bright red to dark red pulsing
            const r = 0.9 + flicker * 0.1;
            const g = 0.05 + Math.sin(timeScale * 2) * 0.05;
            const b = 0.02;
            
            color[i3] = r
            color[i3 + 1] = g
            color[i3 + 2] = b
        }
        
        particlesGeometry.attributes.color.needsUpdate = true
        // Pulsing size for energy
        pointsMaterial.size = 0.10 + Math.sin(timeScale * 5) * 0.03;

    } else if (formationProgress > 0 && currentGesture === 'Blue') {
        const flicker = Math.sin(timeScale * 8) * 0.3 + Math.cos(timeScale * 13) * 0.2;
        const color = particlesGeometry.attributes.color.array

        for (let i = 0; i < count; i ++){
            const i3 = i * 3

            
            // Bright red to dark red pulsing
            const r = flicker * 0.01;
            const g = 0.7 + Math.sin(timeScale * 2) * 0.05;
            const b = 0.8235294118;
            
            color[i3] = r
            color[i3 + 1] = g
            color[i3 + 2] = b
        }  // gojo blue rgb is (69, 176, 210) => (0.2705882353, 0.6901960784, 0.8235294118) now: 0.92 0.05 0.02
        
        particlesGeometry.attributes.color.needsUpdate = true
        // Pulsing size for energy
        pointsMaterial.size = 0.07 + Math.sin(timeScale * 5) * 0.03;

    } else if (isMergeComplete || (formationProgress > 0 && currentGesture == 'Hollow Purple Prepare')) {
        const flicker = Math.sin(timeScale * 8) * 0.3 + Math.cos(timeScale * 13) * 0.2;
        const color = particlesGeometry.attributes.color.array

        for (let i = 0; i < count; i ++){
            const i3 = i * 3;
            const group = particleGroup[i]

            let originalR, originalG, originalB;
            let r, g, b
            const colorLerp = Math.max(0, Math.min(1, (mergeProgress - 0.5) * 2)); // 0 to 1

            if (group === 1){
                originalR = 0.831372549;
                originalG = 0.09803921569;
                originalB = 0.2117647059
            } else if (group === 2){
                originalR = 0.2705882353;
                originalG = 0.6901960784;
                originalB = 0.8235294118;
            }

            const purpleR = 0.5 + flicker * 0.1;
            const purpleG = 0.0;
            const purpleB = 0.5 + flicker * 0.1;

            if (hollowPhase === 'merging') {
                // Purple when merging
                r = originalR + (purpleR - originalR) * colorLerp
                g = originalG + (purpleG - originalG) * colorLerp
                b = originalB + (purpleB - originalB) * colorLerp

            } else if (hollowPhase === 'red_forming') {              // gojo red rgb (212, 25, 54) => (0.831372549, 0.09803921569, 0.2117647059)
                if (group == 1){
                    r = originalR;
                    g = originalG;
                    b = originalB
                }
                else {
                    r = 1; g=1; b=1;
                }
            } else if (hollowPhase === 'blue_forming'){
                if (group == 2){
                    r = originalR;
                    g = originalG;
                    b = originalB
                } else if (group == 1 && redFormationProgress >= 1){
                    r = originalR;
                    g = originalG;
                    b = originalB
                }
            } else {
                r = 1; g=1; b=1;
            }

            if (isMergeComplete || hollowPhase === 'merging') {
                // Force purple if complete, otherwise blend based on merge progress
                const finalLerp = isMergeComplete ? 1 : colorLerp;
                
                r = originalR + (purpleR - originalR) * finalLerp;
                g = originalG + (purpleG - originalG) * finalLerp;
                b = originalB + (purpleB - originalB) * finalLerp;

            } else if (hollowPhase === 'red_forming') {
                if (group == 1) {
                    r = originalR; g = originalG; b = originalB;
                } else {
                    r = 1; g = 1; b = 1; // Unformed particles stay white
                }
            } else if (hollowPhase === 'blue_forming') {
                if (group == 2 || (group == 1 && redFormationProgress >= 1)) {
                    r = originalR; g = originalG; b = originalB;
                } else {
                    r = 1; g = 1; b = 1;
                }
            } else {
                r = 1; g = 1; b = 1; // Default floating state
            }

            color[i3] = r
            color[i3 + 1] = g
            color[i3 + 2] = b
        }

        particlesGeometry.attributes.color.needsUpdate = true;
        pointsMaterial.size = 0.15 + Math.sin(timeScale * 5) * 0.05;

    } else {
        const colors = particlesGeometry.attributes.color.array;
        for (let i = 0; i < count; i++) {
            const i3 = i * 3;
            colors[i3] = 1;
            colors[i3 + 1] = 1;
            colors[i3 + 2] = 1;
        }
        particlesGeometry.attributes.color.needsUpdate = true;
        
        pointsMaterial.size = 0.08;
    }

    particlesGeometry.attributes.position.needsUpdate = true;

    // Camera
    let cameraTargetX, cameraTargetY;
    if (currentHandLandmarks.length > 0 && currentHandLandmarks[0][0]) {
        cameraTargetX = currentHandLandmarks[0][0].x * 0.1;
        cameraTargetY = currentHandLandmarks[0][0].y * 0.1;
    } else {
        cameraTargetX = currentMouse.x * 2;
        cameraTargetY = currentMouse.y * 2;
    }
    
    // Camera shake when sphere is active
    const shake = formationProgress * 0.3;
    camera.position.x += (cameraTargetX - camera.position.x) * 0.02 + (Math.random() - 0.5) * shake;
    camera.position.y += (cameraTargetY - camera.position.y) * 0.02 + (Math.random() - 0.5) * shake;
    camera.lookAt(scene.position);

    renderer.render(scene, camera);
}

// ============================================================
//  DOMAIN EXPANSION — Infinite Void (ANIMATED VERSION)
//  Stage 2: Pink/purple hyperspace tunnel with thin lines
//  Stage 3: Animated circular black hole with left-side effects
// ============================================================

(function initDomainExpansion() {

    // ── INJECT STYLES ────────────────────────────────────────
    const css = document.createElement('style');
    css.textContent = `
        #de-root {
            position: fixed;
            inset: 0;
            z-index: 9999;
            pointer-events: none;
            overflow: hidden;
        }
        #de-flash {
            position: absolute;
            inset: 0;
            background: #ffffff;
            opacity: 0;
            mix-blend-mode: screen;
        }
        #de-tunnel {
            position: absolute;
            inset: 0;
            opacity: 0;
            background: #000;
        }
        #de-tunnel canvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
        }
        #de-void {
            position: absolute;
            inset: 0;
            opacity: 0;
            background: #000000;
        }
        #de-void canvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
        }
    `;
    document.head.appendChild(css);

    // ── BUILD DOM ────────────────────────────────────────────
    const root = document.createElement('div');
    root.id = 'de-root';
    root.innerHTML = `
        <div id="de-tunnel"><canvas id="de-tunnel-canvas"></canvas></div>
        <div id="de-void"><canvas id="de-void-canvas"></canvas></div>
        <div id="de-flash"></div>
    `;
    document.body.appendChild(root);

    const flashEl      = document.getElementById('de-flash');
    const tunnelEl     = document.getElementById('de-tunnel');
    const tunnelCanvas = document.getElementById('de-tunnel-canvas');
    const voidEl       = document.getElementById('de-void');
    const voidCanvas   = document.getElementById('de-void-canvas');

    // ── CANVAS RESIZE ────────────────────────────────────────
    function resizeAll() {
        tunnelCanvas.width  = window.innerWidth;
        tunnelCanvas.height = window.innerHeight;
        voidCanvas.width    = window.innerWidth;
        voidCanvas.height   = window.innerHeight;
    }
    resizeAll();
    window.addEventListener('resize', resizeAll);

    const tCtx = tunnelCanvas.getContext('2d');
    const vCtx = voidCanvas.getContext('2d');

    // ── STATE ────────────────────────────────────────────────
    let dePhase    = 'idle';
    let deTimer    = 0;
    let holdTimer  = 0;
    let lastTime   = 0;
    let tunnelTime = 0;
    let voidTime   = 0;

    const FLASH_DURATION  = 1.5;
    const TUNNEL_DURATION = 3.0;

    // ── STAGE 2: PINK/PURPLE HYPERSPACE TUNNEL ───────────────
    const LINE_COUNT = 400;
    const tunnelLines = [];
    
    const colors = [
        [255, 0, 128], [255, 0, 255], [200, 0, 255],
        [150, 0, 200], [255, 100, 200], [180, 50, 220],
    ];
    
    for (let i = 0; i < LINE_COUNT; i++) {
        const angle = (i / LINE_COUNT) * Math.PI * 2 + (Math.random() - 0.5) * 0.5;
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        tunnelLines.push({
            angle: angle,
            width: Math.random() * 1.5 + 0.3,
            color: color,
            speed: Math.random() * 2 + 1.5,
            offset: Math.random() * 100,
            length: Math.random() * 0.3 + 0.1,
            wobble: Math.random() * 0.02
        });
    }

    const PARTICLE_COUNT = 200;
    const tunnelParticles = [];
    
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const color = colors[Math.floor(Math.random() * colors.length)];
        tunnelParticles.push({
            angle: Math.random() * Math.PI * 2,
            distance: Math.random() * 2000 + 100,
            speed: Math.random() * 15 + 10,
            size: Math.random() * 2 + 0.5,
            color: color,
            brightness: Math.random() * 0.5 + 0.5
        });
    }

    function drawTunnel(progress) {
        const W = tunnelCanvas.width;
        const H = tunnelCanvas.height;
        const cx = W / 2;
        const cy = H / 2;
        
        const bgGrad = tCtx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(W, H));
        bgGrad.addColorStop(0, '#1a001a');
        bgGrad.addColorStop(0.4, '#0d000d');
        bgGrad.addColorStop(1, '#000000');
        tCtx.fillStyle = bgGrad;
        tCtx.fillRect(0, 0, W, H);

        const speed = Math.pow(progress, 0.4) * 3;
        const zoom = 1 + progress * 5;

        tCtx.save();
        tCtx.translate(cx, cy);
        
        for (const line of tunnelLines) {
            const time = (tunnelTime * line.speed + line.offset) % 100;
            const t = time / 100;
            
            const innerR = t * 20;
            const outerR = t * Math.max(W, H) * 2.5;
            
            if (outerR <= innerR) continue;
            
            const wobble = Math.sin(t * Math.PI * 4 + tunnelTime * line.wobble) * 0.02;
            const angle = line.angle + wobble;
            
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            
            const x1 = cos * innerR;
            const y1 = sin * innerR;
            const x2 = cos * outerR;
            const y2 = sin * outerR;
            
            const streakStart = Math.max(0, t - line.length * (1 + speed * 0.5));
            const sx = cos * (innerR + (outerR - innerR) * streakStart);
            const sy = sin * (innerR + (outerR - innerR) * streakStart);
            
            const alpha = (1 - t) * line.width * 0.4 * speed;
            const [r, g, b] = line.color;
            
            const grad = tCtx.createLinearGradient(sx, sy, x2, y2);
            grad.addColorStop(0, `rgba(${r},${g},${b},0)`);
            grad.addColorStop(0.3, `rgba(${r},${g},${b},${alpha * 0.3})`);
            grad.addColorStop(0.7, `rgba(255,255,255,${alpha * 0.8})`);
            grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
            
            tCtx.beginPath();
            tCtx.moveTo(sx, sy);
            tCtx.lineTo(x2, y2);
            tCtx.strokeStyle = grad;
            tCtx.lineWidth = line.width * (0.5 + speed);
            tCtx.stroke();
        }
        
        for (const p of tunnelParticles) {
            p.distance -= p.speed * speed;
            if (p.distance < 10) {
                p.distance = 2000;
                p.angle = Math.random() * Math.PI * 2;
            }
            
            const perspective = 100 / (100 + p.distance);
            const x = Math.cos(p.angle) * p.distance * perspective * zoom;
            const y = Math.sin(p.angle) * p.distance * perspective * zoom;
            
            const size = p.size * perspective * zoom * (0.5 + speed);
            const alpha = p.brightness * perspective * speed;
            const [r, g, b] = p.color;
            
            const streakLen = speed * 20 * perspective;
            const dx = Math.cos(p.angle) * streakLen;
            const dy = Math.sin(p.angle) * streakLen;
            
            const grad = tCtx.createLinearGradient(x - dx, y - dy, x, y);
            grad.addColorStop(0, `rgba(${r},${g},${b},0)`);
            grad.addColorStop(1, `rgba(255,255,255,${alpha})`);
            
            tCtx.beginPath();
            tCtx.moveTo(x - dx, y - dy);
            tCtx.lineTo(x, y);
            tCtx.strokeStyle = grad;
            tCtx.lineWidth = size;
            tCtx.stroke();
        }
        
        tCtx.restore();

        const coreSize = 20 + speed * 100;
        const coreGrad = tCtx.createRadialGradient(cx, cy, 0, cx, cy, coreSize);
        coreGrad.addColorStop(0, `rgba(255,255,255,${0.9 * speed})`);
        coreGrad.addColorStop(0.1, `rgba(255,200,255,${0.6 * speed})`);
        coreGrad.addColorStop(0.3, `rgba(255,0,200,${0.3 * speed})`);
        coreGrad.addColorStop(0.6, `rgba(100,0,150,${0.1 * speed})`);
        coreGrad.addColorStop(1, 'rgba(0,0,0,0)');
        
        tCtx.fillStyle = coreGrad;
        tCtx.beginPath();
        tCtx.arc(cx, cy, coreSize, 0, Math.PI * 2);
        tCtx.fill();

        const vigGrad = tCtx.createRadialGradient(cx, cy, coreSize * 0.5, cx, cy, Math.max(W, H));
        vigGrad.addColorStop(0, 'rgba(0,0,0,0)');
        vigGrad.addColorStop(0.5, `rgba(0,0,0,${0.2 * speed})`);
        vigGrad.addColorStop(1, `rgba(0,0,0,${0.8 * speed})`);
        tCtx.fillStyle = vigGrad;
        tCtx.fillRect(0, 0, W, H);
    }

    // ── STAGE 3: ANIMATED CIRCULAR BLACK HOLE ────────────────
    // With left-side effects and subtle animations
    
    let voidAnimationId = null;
    
    // Pre-generate static particles
    const DISK_COUNT = 600;
    const diskParticles = [];
    const leftParticles = []; // For left side
    
    function generateVoidParticles(W, H, cx, cy, holeR, ringInner, ringOuter) {
        diskParticles.length = 0;
        leftParticles.length = 0;
        
        // Main accretion disk
        for (let i = 0; i < DISK_COUNT; i++) {
            const angle = Math.random() * Math.PI * 2;
            const radius = 0.4 + Math.abs((Math.random() - 0.5) * 2) * 0.5;
            const r = ringInner + (ringOuter - ringInner) * radius;
            
            const x = Math.cos(angle) * r;
            const y = Math.sin(angle) * r;
            
            const doppler = Math.cos(angle);
            const isFront = Math.sin(angle) > 0;
            const temp = Math.pow(1 - Math.abs(radius - 0.5) * 2, 0.5);
            
            let rr, gg, bb;
            if (doppler > 0) {
                rr = 200 + temp * 55; gg = 230; bb = 255;
            } else {
                rr = 255; gg = 200 + temp * 55; bb = 180 + temp * 75;
            }
            
            const brightness = (0.4 + temp * 0.6) * (1 + Math.abs(doppler) * 0.4);
            const size = (0.8 + temp * 1.5) * (0.8 + Math.random() * 0.4);
            const alpha = (0.3 + Math.random() * 0.5) * brightness;
            
            diskParticles.push({
                x, y, rr, gg, bb, alpha, size, isFront, angle, r,
                baseAlpha: alpha, pulseSpeed: 0.5 + Math.random() * 1.5
            });
        }
        
        diskParticles.sort((a, b) => {
            if (a.isFront !== b.isFront) return a.isFront ? 1 : -1;
            return a.r - b.r;
        });
        
        // Left side particles (incoming matter)
        const LEFT_COUNT = 1500;
        for (let i = 0; i < LEFT_COUNT; i++) {
            const angle = Math.PI + (Math.random() - 0.5) * 1.5; // Left side angles
            const distance = holeR * 2 + Math.random() * holeR * 3;
            const spread = (Math.random() - 0.5) * holeR * 0.5;
            
            leftParticles.push({
                angle: angle,
                distance: distance,
                spread: spread,
                size: Math.random() * 2 + 0.5,
                speed: 0.5 + Math.random() * 1.5,
                alpha: Math.random() * 0.6 + 0.2,
                color: Math.random() > 0.5 ? [200, 220, 255] : [255, 220, 200]
            });
        }
    }

    function animateVoid() {
        const W = voidCanvas.width;
        const H = voidCanvas.height;
        const cx = W * 0.5;
        const cy = H * 0.5;
        const minDim = Math.min(W, H);
        
        const holeR = minDim * 0.35;
        const photonR = holeR * 1.06;
        const ringInner = holeR * 1.12;
        const ringOuter = holeR * 1.9;
        
        // Generate particles once
        if (diskParticles.length === 0) {
            generateVoidParticles(W, H, cx, cy, holeR, ringInner, ringOuter);
        }
        
        // Clear
        vCtx.fillStyle = '#000000';
        vCtx.fillRect(0, 0, W, H);
        
        // Background
        const bgGrad = vCtx.createRadialGradient(cx, cy, holeR * 3, cx, cy, minDim);
        bgGrad.addColorStop(0, 'rgba(20,20,40,0.2)');
        bgGrad.addColorStop(0.5, 'rgba(10,10,20,0.1)');
        bgGrad.addColorStop(1, 'rgba(0,0,0,0)');
        vCtx.fillStyle = bgGrad;
        vCtx.fillRect(0, 0, W, H);

        const t = voidTime * 0.001; // Slow time scale

        // Draw left side incoming matter (animated)
        vCtx.save();
        vCtx.translate(cx, cy);
        
        for (const p of leftParticles) {
            // Animate spiral inward
            p.distance -= p.speed * 2;
            p.angle += 0.02 * p.speed;
            
            if (p.distance < ringInner) {
                p.distance = holeR * 3.5;
                p.angle = Math.PI + (Math.random() - 0.5) * 1.5;
            }
            
            const x = Math.cos(p.angle) * p.distance;
            const y = Math.sin(p.angle) * p.distance + p.spread * Math.sin(t * 2 + p.distance * 0.01);
            
            const fadeIn = Math.min(1, (holeR * 3.5 - p.distance) / (holeR));
            const twinkle = 0.7 + Math.sin(t * p.speed * 3) * 0.3;
            const alpha = p.alpha * fadeIn * twinkle;
            
            const [r, g, b] = p.color;
            
            // Draw particle
            vCtx.beginPath();
            vCtx.arc(x, y, p.size, 0, Math.PI * 2);
            vCtx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
            vCtx.fill();
            
            // Trail
            const trailLen = p.speed * 10;
            const tx = x - Math.cos(p.angle) * trailLen;
            const ty = y - Math.sin(p.angle) * trailLen;
            
            const grad = vCtx.createLinearGradient(tx, ty, x, y);
            grad.addColorStop(0, `rgba(${r},${g},${b},0)`);
            grad.addColorStop(1, `rgba(${r},${g},${b},${alpha * 0.5})`);
            
            vCtx.beginPath();
            vCtx.moveTo(tx, ty);
            vCtx.lineTo(x, y);
            vCtx.strokeStyle = grad;
            vCtx.lineWidth = p.size * 0.5;
            vCtx.stroke();
        }
        
        vCtx.restore();

        // Draw back half of disk (with subtle pulse)
        vCtx.save();
        vCtx.translate(cx, cy);
        
        for (const p of diskParticles) {
            if (p.isFront) continue;
            
            // Subtle pulsing
            const pulse = 0.9 + Math.sin(t * p.pulseSpeed * 2) * 0.1;
            const alpha = p.baseAlpha * pulse;
            
            const glow = vCtx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 1.5);
            glow.addColorStop(0, `rgba(${p.rr},${p.gg},${p.bb},${alpha * 0.6})`);
            glow.addColorStop(1, `rgba(${p.rr},${p.gg},${p.bb},0)`);
            
            vCtx.fillStyle = glow;
            vCtx.beginPath();
            vCtx.arc(p.x, p.y, p.size * 1.5, 0, Math.PI * 2);
            vCtx.fill();
        }
        
        vCtx.restore();

        // Photon ring (with subtle rotation)
        vCtx.save();
        vCtx.translate(cx, cy);
        vCtx.rotate(t * 0.1); // Very slow rotation
        
        const photonGrad = vCtx.createRadialGradient(0, 0, photonR * 0.98, 0, 0, photonR * 1.04);
        photonGrad.addColorStop(0, 'rgba(255,255,255,0)');
        photonGrad.addColorStop(0.5, 'rgba(255,255,255,0.95)');
        photonGrad.addColorStop(0.7, 'rgba(255,250,220,0.7)');
        photonGrad.addColorStop(0.9, 'rgba(200,220,255,0.3)');
        photonGrad.addColorStop(1, 'rgba(255,255,255,0)');
        
        vCtx.fillStyle = photonGrad;
        vCtx.beginPath();
        vCtx.arc(0, 0, photonR * 1.04, 0, Math.PI * 2);
        vCtx.fill();
        
        vCtx.restore();

        // Event horizon (subtle breathing)
        vCtx.save();
        vCtx.translate(cx, cy);
        
        const breatheR = holeR * (1 + Math.sin(t * 0.5) * 0.005);
        
        const edgeGrad = vCtx.createRadialGradient(0, 0, breatheR * 0.96, 0, 0, breatheR * 1.02);
        edgeGrad.addColorStop(0, 'rgba(0,0,0,1)');
        edgeGrad.addColorStop(0.9, 'rgba(0,0,0,1)');
        edgeGrad.addColorStop(0.95, 'rgba(50,40,80,0.6)');
        edgeGrad.addColorStop(1, 'rgba(100,90,150,0.2)');
        
        vCtx.fillStyle = edgeGrad;
        vCtx.beginPath();
        vCtx.arc(0, 0, breatheR * 1.02, 0, Math.PI * 2);
        vCtx.fill();
        
        vCtx.fillStyle = '#000000';
        vCtx.beginPath();
        vCtx.arc(0, 0, breatheR * 0.98, 0, Math.PI * 2);
        vCtx.fill();
        
        vCtx.restore();

        // Draw front half of disk (with pulse)
        vCtx.save();
        vCtx.translate(cx, cy);
        
        for (const p of diskParticles) {
            if (!p.isFront) continue;
            
            const pulse = 0.9 + Math.sin(t * p.pulseSpeed * 2) * 0.1;
            const alpha = p.baseAlpha * pulse;
            
            vCtx.beginPath();
            vCtx.arc(p.x, p.y, p.size * 0.5, 0, Math.PI * 2);
            vCtx.fillStyle = `rgba(255,255,255,${alpha})`;
            vCtx.fill();
            
            const glow = vCtx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 2);
            glow.addColorStop(0, `rgba(${p.rr},${p.gg},${p.bb},${alpha * 0.9})`);
            glow.addColorStop(1, `rgba(${p.rr},${p.gg},${p.bb},0)`);
            
            vCtx.fillStyle = glow;
            vCtx.beginPath();
            vCtx.arc(p.x, p.y, p.size * 2, 0, Math.PI * 2);
            vCtx.fill();
        }
        
        vCtx.restore();

        // Energy jet (animated)
        const JET_COUNT = 20;
        vCtx.save();
        vCtx.translate(cx, cy);
        
        for (let i = 0; i < JET_COUNT; i++) {
            const progress = (i / JET_COUNT - t * 0.00001) % 1;
            const x = ringOuter * 0.95 + progress * minDim * 0.6;
            const spread = Math.sin(progress * Math.PI) * holeR * 0.4;
            const y = (Math.random() - 0.5) * spread * 2 + Math.sin(t * 2 + progress * 10) * 5;
            
            const size = (1.5 + progress * 6) * (0.5 + Math.random());
            const alpha = (0.2 + Math.random() * 0.5) * (1 - progress * 0.6) * (0.8 + Math.sin(t * 3 + i) * 0.2);
            
            const streakLen = 20 + progress * 80;
            
            const grad = vCtx.createLinearGradient(x - streakLen, y, x, y);
            grad.addColorStop(0, `rgba(200,220,255,0)`);
            grad.addColorStop(0.5, `rgba(220,240,255,${alpha * 0.4})`);
            grad.addColorStop(1, `rgba(255,255,255,${alpha})`);
            
            vCtx.beginPath();
            vCtx.moveTo(x - streakLen, y);
            vCtx.lineTo(x, y);
            vCtx.strokeStyle = grad;
            vCtx.lineWidth = size * 0.5;
            vCtx.stroke();
            
            vCtx.beginPath();
            vCtx.arc(x, y, size * 0.5, 0, Math.PI * 2);
            vCtx.fillStyle = `rgba(255,255,255,${alpha * 0.9})`;
            vCtx.fill();
        }
        
        vCtx.restore();

        // Gravitational lensing halo (pulsing)
        const lensGrad = vCtx.createRadialGradient(cx, cy, photonR, cx, cy, ringOuter * 1.3);
        const lensAlpha = 0.1 + Math.sin(t) * 0.02;
        lensGrad.addColorStop(0, 'rgba(255,255,255,0)');
        lensGrad.addColorStop(0.4, `rgba(255,255,240,${lensAlpha})`);
        lensGrad.addColorStop(0.7, `rgba(220,230,255,${lensAlpha * 0.5})`);
        lensGrad.addColorStop(1, 'rgba(0,0,0,0)');
        
        vCtx.fillStyle = lensGrad;
        vCtx.beginPath();
        vCtx.arc(cx, cy, ringOuter * 1.3, 0, Math.PI * 2);
        vCtx.fill();

        voidTime += 16; // ~60fps
        voidAnimationId = requestAnimationFrame(animateVoid);
    }

    // ── HELPERS ──────────────────────────────────────────────
    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    function easeIn(t) { return t * t * t; }
    function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

    function setWebcamFilter(f) {
        const vid = document.getElementById('webcam');
        if (vid) vid.style.filter = f;
    }

    function resetAll() {
        dePhase = 'idle';
        deTimer = 0;
        holdTimer = 0;
        tunnelTime = 0;
        voidTime = 0;
        
        if (voidAnimationId) {
            cancelAnimationFrame(voidAnimationId);
            voidAnimationId = null;
        }
        
        diskParticles.length = 0;
        leftParticles.length = 0;
        
        flashEl.style.opacity = '0';
        tunnelEl.style.opacity = '0';
        voidEl.style.opacity = '0';
        voidEl.style.transform = 'scale(1)';
        
        setWebcamFilter('');
        tCtx.clearRect(0, 0, tunnelCanvas.width, tunnelCanvas.height);
        vCtx.clearRect(0, 0, voidCanvas.width, voidCanvas.height);
    }

    // ── MAIN LOOP ────────────────────────────────────────────
    function deLoop(now) {
        requestAnimationFrame(deLoop);

        const dt = lastTime ? clamp((now - lastTime) / 1000, 0, 0.05) : 0;
        lastTime = now;

        const gesture = typeof currentGesture !== 'undefined' ? currentGesture : null;

        // ── HOLD CHARGE (REMOVED VISUAL RING) ─────────────────
        if (dePhase === 'idle') {
            if (gesture === 'Domain Expansion') {
                holdTimer += dt;
                if (holdTimer >= 1.5) { // Hardcoded hold time, no visual
                    dePhase = 'flash';
                    deTimer = 0;
                    holdTimer = 0;
                }
            } else {
                holdTimer = Math.max(0, holdTimer - dt * 2);
            }
            return;
        }

        if (gesture !== 'Domain Expansion' && dePhase === 'void') {
            dePhase = 'collapsing';
            deTimer = 0;
        }

        deTimer += dt;

        // ── PHASE: FLASH ──────────────────────────────────────
        if (dePhase === 'flash') {
            const t = deTimer / FLASH_DURATION;
            
            let op;
            if (t < 0.05) {
                op = easeOut(t / 0.05);
            } else if (t < 0.15) {
                op = 1;
            } else if (t < 0.6) {
                const decay = (t - 0.15) / 0.45;
                op = Math.max(0, 1 - easeIn(decay));
            } else {
                op = Math.max(0, 0.1 - (t - 0.6) / 0.4 * 0.1);
            }
            
            op = clamp(op, 0, 1);
            flashEl.style.opacity = op;
            
            const brightness = 1 - op * 0.98;
            setWebcamFilter(`brightness(${brightness}) saturate(${1 - op * 0.9})`);

            if (deTimer >= FLASH_DURATION) {
                dePhase = 'tunnel';
                deTimer = 0;
                tunnelTime = 0;
                flashEl.style.opacity = '0';
                tunnelEl.style.opacity = '1';
                setWebcamFilter('brightness(0)');
            }
        }

        // ── PHASE: TUNNEL ─────────────────────────────────────
        if (dePhase === 'tunnel') {
            tunnelTime += dt;
            const progress = clamp(deTimer / TUNNEL_DURATION, 0, 1);
            drawTunnel(progress);

            if (deTimer >= TUNNEL_DURATION) {
                dePhase = 'void';
                deTimer = 0;
                voidTime = 0;
                tunnelEl.style.opacity = '0';
                voidEl.style.opacity = '1';
                animateVoid(); // Start animation loop
            }
        }

        // ── PHASE: VOID ───────────────────────────────────────
        if (dePhase === 'void') {
            // Animation handled by animateVoid
        }

        // ── PHASE: COLLAPSING ─────────────────────────────────
        if (dePhase === 'collapsing') {
            const dur = 1.0;
            const t = clamp(deTimer / dur, 0, 1);

            voidEl.style.opacity = String(1 - easeIn(t));
            
            if (t > 0.1 && t < 0.3) {
                const flashT = (t - 0.1) / 0.2;
                flashEl.style.opacity = String(Math.sin(flashT * Math.PI) * 0.5);
            } else {
                flashEl.style.opacity = '0';
            }

            setWebcamFilter(`brightness(${clamp(t, 0, 1)})`);

            if (deTimer >= dur) {
                resetAll();
            }
        }
    }
    requestAnimationFrame(deLoop);

})();

animate()