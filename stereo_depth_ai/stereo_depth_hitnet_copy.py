import depthai as dai
import cv2
import numpy as np
import time
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
    # --- 1. Setup Center RGB Camera (CAM_A) ---
    camCenter = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    # Changed FPS to 30 to respect hardware limits
    outputCenter = camCenter.requestOutput((640, 400), type=dai.ImgFrame.Type.BGR888i, fps=30)

    # --- 2. Setup Left/Right Mono Cameras (CAM_B & CAM_C) ---
    camLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    camRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    outputLeft = camLeft.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=30)
    outputRight = camRight.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=30)

    # --- 3. Create StereoDepth node ---
    stereo = pipeline.create(dai.node.StereoDepth)
    outputLeft.link(stereo.left)
    outputRight.link(stereo.right)

    # --- 4. Create Output Queues (DepthAI V3 Native) ---
    qCenterRGB = outputCenter.createOutputQueue(maxSize=1, blocking=True)
    qLeftRect = stereo.rectifiedLeft.createOutputQueue(maxSize=1, blocking=True)
    qRightRect = stereo.rectifiedRight.createOutputQueue(maxSize=1, blocking=True)

    pipeline.start()
    pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(0.8)

    startTime = time.monotonic()
    counter = 0
    fps = 0.0
    run_inference = True 

    print("Pipeline started.")
    print("Press 's' to save a dataset pair.")
    print("Press 'i' to toggle HITNET inference.")
    print("Press 'q' to quit.")

    while pipeline.isRunning():
        # Pull frames from all three queues
        inCenterRGB = qCenterRGB.get()
        inLeftRect = qLeftRect.get()
        inRightRect = qRightRect.get()

        # Extract OpenCV matrices
        frameCenterRGB = inCenterRGB.getCvFrame() # This is your True Color image!
        frameLeftMono = inLeftRect.getCvFrame()
        frameRightMono = inRightRect.getCvFrame()

        if run_inference:
            # HitNet still needs the Left/Right rectified frames
            # Convert them from 1-channel mono to 3-channel dummy BGR for the neural network
            left_bgr = cv2.cvtColor(frameLeftMono, cv2.COLOR_GRAY2BGR)
            right_bgr = cv2.cvtColor(frameRightMono, cv2.COLOR_GRAY2BGR)

            infer_w, infer_h = 320, 200
            left_infer = cv2.resize(left_bgr, (infer_w, infer_h))
            right_infer = cv2.resize(right_bgr, (infer_w, infer_h))

            # Run Inference
            disparity_map = hitnet_depth(left_infer, right_infer)
            
            # Normalize and Colorize the Disparity Map
            disp_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_disparity = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
            color_disparity_disp = cv2.resize(color_disparity, (640, 400))

            # Display the TRUE RGB image next to the HitNet disparity map
            combined_view = cv2.hconcat([frameCenterRGB, color_disparity_disp])
            
            cv2.putText(combined_view, "Inference: ON", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
        else:
            # If inference is off, just show the Center RGB frame
            combined_view = frameCenterRGB.copy()
            cv2.putText(combined_view, "Inference: OFF", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS Calculation
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1.0:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        
        cv2.putText(combined_view, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Data Collection", combined_view)

        key = cv2.waitKey(1) & 0xFF
        
        # Save Dataset
        if key == ord('s'):
            rgb_name = os.path.join(dataset_dir, f"color_{saved_pairs_count:04d}.png")
            left_name = os.path.join(dataset_dir, f"left_{saved_pairs_count:04d}.png")
            right_name = os.path.join(dataset_dir, f"right_{saved_pairs_count:04d}.png")
            
            # Save all three so you have the color reference AND the perfect stereo pair
            cv2.imwrite(rgb_name, frameCenterRGB)
            cv2.imwrite(left_name, frameLeftMono)
            cv2.imwrite(right_name, frameRightMono)
            
            print(f"Saved trio #{saved_pairs_count} to {dataset_dir} (Center RGB, Left Mono, Right Mono)")
            saved_pairs_count += 1
            
        elif key == ord('i'):
            run_inference = not run_inference
            print(f"HITNET Inference is now {'ON' if run_inference else 'OFF'}")
            
        elif key == ord('q'):
            break

cv2.destroyAllWindows()