import depthai as dai
import cv2
import numpy as np
import time
from hitnet import HitNet, ModelType, draw_disparity
import os

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

    # Request output
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
    pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(0.8)

    # FPS Variables Setup
    startTime = time.monotonic()
    counter = 0
    fps = 0.0

    print("Pipeline started. Press 's' to save a dataset pair. Press 'q' to quit.")

    while pipeline.isRunning():
        # Pull frames
        inLeft = qLeft.get()
        inRight = qRight.get()

        # Convert to OpenCV format
        frameLeft = inLeft.getCvFrame()
        frameRight = inRight.getCvFrame()

        # Grayscale
        left_bgr = cv2.cvtColor(frameLeft, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(frameRight, cv2.COLOR_GRAY2BGR)

        left_bgr = cv2.resize(left_bgr, (640, 480))
        right_bgr = cv2.resize(right_bgr, (640, 480))

        # Run Hitnet Inference
        disparity_map = hitnet_depth(left_bgr, right_bgr)

        color_disparity = draw_disparity(disparity_map)

        combined_view = cv2.hconcat([left_bgr, color_disparity])

        # Calculate and draw FPS
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1.0:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        
        cv2.putText(combined_view, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live HITNET Inference", combined_view)

        key = cv2.waitKey(1) & 0xFF
        
        # Dataset Collection
        if key == ord('s'):
            # Save the pure, raw GRAY8 frames (not the BGR trick ones)
            left_name = os.path.join(dataset_dir, f"left_{saved_pairs_count:04d}.png")
            right_name = os.path.join(dataset_dir, f"right_{saved_pairs_count:04d}.png")
            
            cv2.imwrite(left_name, frameLeft)
            cv2.imwrite(right_name, frameRight)
            print(f"Saved pair #{saved_pairs_count} to {dataset_dir}")
            saved_pairs_count += 1
            
        elif key == ord('q'):
            break

cv2.destroyAllWindows()