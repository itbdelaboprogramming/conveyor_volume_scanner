import depthai as dai
import cv2
import numpy as np
import time
# Removed draw_disparity from import since we are using custom OpenCV colormaps now
from hitnet import HitNet, ModelType
import os
import tensorflow as tf

# 1. Enable Mixed Precision to speed up RTX 3050 operations
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# create folder for Hailo calibration dataset
dataset_dir = "calibration_dataset"
os.makedirs(dataset_dir, exist_ok=True)
saved_pairs_count = 0

# Setup HITNET model
model_path = "models/eth3d.pb"
hitnet_depth = HitNet(model_path, ModelType.eth3d)

device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)

with dai.Pipeline(device) as pipeline:
    # Create camera nodes
    camLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    camRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Request output (Native 640x400)
    outputLeftRaw = camLeft.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=60)
    outputRightRaw = camRight.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=60)

    # Create StereoDepth node
    stereo = pipeline.create(dai.node.StereoDepth)
    outputLeftRaw.link(stereo.left)
    outputRightRaw.link(stereo.right)

    # Create output queues
    qLeft = stereo.rectifiedLeft.createOutputQueue(maxSize=1, blocking=True)
    qRight = stereo.rectifiedRight.createOutputQueue(maxSize=1, blocking=True)

    # Start the pipeline
    pipeline.start()

    # Turn on the IR Projector
    # pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(0.8)

    # FPS Variables Setup
    startTime = time.monotonic()
    counter = 0
    fps = 0.0
    
    # 2. Toggle for inference
    run_inference = True 

    print("Pipeline started.")
    print("Press 's' to save a dataset pair.")
    print("Press 'i' to toggle HITNET inference ON/OFF (turn OFF for fast dataset collection).")
    print("Press 'q' to quit.")

    while pipeline.isRunning():
        # Pull frames
        inLeft = qLeft.get()
        inRight = qRight.get()

        # Convert to OpenCV format
        frameLeft = inLeft.getCvFrame()
        frameRight = inRight.getCvFrame()

        # Grayscale to BGR for display/inference
        left_bgr = cv2.cvtColor(frameLeft, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(frameRight, cv2.COLOR_GRAY2BGR)

        if run_inference:
            # 3. Downscale JUST for the inference step to boost FPS (maintain aspect ratio)
            infer_w, infer_h = 320, 200
            left_infer = cv2.resize(left_bgr, (infer_w, infer_h))
            right_infer = cv2.resize(right_bgr, (infer_w, infer_h))

            # Run Hitnet Inference
            disparity_map = hitnet_depth(left_infer, right_infer)
            
            # --- NEW COLORMAP LOGIC ---
            # Normalize the raw float disparity map to 0-255 integers
            disp_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply the Heat-Cool (JET) colormap
            # Tip: You can also try cv2.COLORMAP_TURBO for a more modern, perceptually uniform alternative
            color_disparity = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
            # --------------------------

            # Resize the disparity map back up so we can display it next to the original
            color_disparity_disp = cv2.resize(color_disparity, (640, 400))
            combined_view = cv2.hconcat([left_bgr, color_disparity_disp])
            
            # Add a status text
            cv2.putText(combined_view, "Inference: ON", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
        else:
            # If inference is off, just show the left frame to save processing power
            combined_view = left_bgr.copy()
            cv2.putText(combined_view, "Inference: OFF (Fast Capture Mode)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Calculate and draw FPS
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1.0:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        
        cv2.putText(combined_view, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Data Collection", combined_view)

        key = cv2.waitKey(1) & 0xFF
        
        # Dataset Collection
        if key == ord('s'):
            # Save the pure, raw GRAY8 frames natively at 640x400
            left_name = os.path.join(dataset_dir, f"left_{saved_pairs_count:04d}.png")
            right_name = os.path.join(dataset_dir, f"right_{saved_pairs_count:04d}.png")
            
            cv2.imwrite(left_name, frameLeft)
            cv2.imwrite(right_name, frameRight)
            print(f"Saved pair #{saved_pairs_count} to {dataset_dir} (640x400)")
            saved_pairs_count += 1
            
        # Toggle Inference
        elif key == ord('i'):
            run_inference = not run_inference
            print(f"HITNET Inference is now {'ON' if run_inference else 'OFF'}")
            
        elif key == ord('q'):
            break

cv2.destroyAllWindows()