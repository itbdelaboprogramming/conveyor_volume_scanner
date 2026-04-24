import depthai as dai
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from hitnet import HitNet, ModelType, draw_disparity

# Enable Mixed Precision for RTX 3050 speedup
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Setup HITNET model
model_path = "models/eth3d.pb"
hitnet_depth = HitNet(model_path, ModelType.eth3d)

# --- System & Math Constants ---
DISPLAY_W = 800
DISPLAY_H = 600
RAW_BOX_SIZE = 400
raw_half_box = RAW_BOX_SIZE // 2  
MAX_DISTANCE_MM = 3000 

# Stereo to Depth Constants (OAK-D specific)
OAK_BASELINE_MM = 75.0  # Distance between Left/Right lenses
FOCAL_LENGTH_PIXELS = 200.0 # Approx focal length for a 320x200 resized image

# Volume Calculation Constants
BASELINE_MM = 743.0  # Distance to empty conveyor belt in mm
MM_PER_PIXEL = 2.78    # Physical width/height of 1 pixel at the baseline distance
PIXEL_AREA_MM2 = MM_PER_PIXEL ** 2
HEIGHT_THRESHOLD = 15.0

# --- Matplotlib 3D Setup ---
plt.ion() 
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(RAW_BOX_SIZE), np.arange(RAW_BOX_SIZE))

# --- Helper Function for Legend ---
def draw_detailed_legend(frame, max_distance_mm, step_size=500):
    frame_h, frame_w = frame.shape[:2]
    legend_w, legend_h = 35, int(frame_h * 0.85) 
    
    start_x = frame_w - legend_w - 110 
    start_y = (frame_h - legend_h) // 2
    
    gradient_1d = np.linspace(255, 0, legend_h).astype(np.uint8)
    gradient_2d = np.tile(gradient_1d, (legend_w, 1)).T
    legend_colored = cv2.applyColorMap(gradient_2d, cv2.COLORMAP_JET)
    
    frame[start_y : start_y + legend_h, start_x : start_x + legend_w] = legend_colored
    cv2.rectangle(frame, (start_x, start_y), (start_x + legend_w, start_y + legend_h), (255, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    num_steps = max_distance_mm // step_size
    for i in range(num_steps + 1):
        val = i * step_size
        y_pos = start_y + int((i / num_steps) * legend_h)
        cv2.line(frame, (start_x + legend_w, y_pos), (start_x + legend_w + 10, y_pos), (255, 255, 255), 2)
        text_y = y_pos + 5
        if i == 0: text_y += 8
        elif i == num_steps: text_y -= 5
        text = f"{val}mm" if i < num_steps else f">{val}mm"
        cv2.putText(frame, text, (start_x + legend_w + 15, text_y), font, 0.6, (255, 255, 255), 2)
    return frame

# --- Main Setup (DepthAI v3 API) ---
print("Connecting to OAK-D Pro...")
device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)

with dai.Pipeline(device) as pipeline:
    # Set up Left/Right Mono Cameras
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    leftOut = monoLeft.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=60)

    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rightOut = monoRight.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=60)

    # Set up Color Camera
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgbOut = camRgb.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i)

    # We still use StereoDepth to get the perfectly rectified pairs
    stereo = pipeline.create(dai.node.StereoDepth)
    leftOut.link(stereo.left)
    rightOut.link(stereo.right)

    # Output Queues
    qLeft = stereo.rectifiedLeft.createOutputQueue(maxSize=2, blocking=False)
    qRight = stereo.rectifiedRight.createOutputQueue(maxSize=2, blocking=False)
    qRgb = rgbOut.createOutputQueue(maxSize=2, blocking=False)

    print("Starting pipeline...")
    pipeline.start()

    # Turn on IR Laser
    try:
        device.setIrLaserDotProjectorBrightness(1200)
    except Exception as e:
        print(f"Note: Could not set IR Laser dot projector. ({e})")

    window_name = "OAK-D AI Conveyor Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W * 2, DISPLAY_H)

    startTime = time.monotonic()
    depth_frame_count = 0 
    total_frames = 0
    fps = 0

    latest_rgb_display = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    latest_depth_display = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    volume_cm3 = 0.0

    print("Dashboard and Integration started! Press 'q' to quit.")
    print("Press 't' to TARE the baseline.")

    try:
        while pipeline.isRunning():
            # Pull frames
            inLeft = qLeft.tryGet()
            inRight = qRight.tryGet()
            inRgb = qRgb.tryGet()

            currentTime = time.monotonic()
            if (currentTime - startTime) > 1: 
                fps = depth_frame_count / (currentTime - startTime)
                depth_frame_count = 0
                startTime = currentTime

            # --- 1. Process RGB ---
            if inRgb is not None:
                rgb_frame = inRgb.getCvFrame() 
                rgb_resized = cv2.resize(rgb_frame, (DISPLAY_W, DISPLAY_H))
                scale_x, scale_y = DISPLAY_W / rgb_frame.shape[1], DISPLAY_H / rgb_frame.shape[0]
                draw_w, draw_h = int(raw_half_box * scale_x), int(raw_half_box * scale_y)
                
                cv2.rectangle(rgb_resized, (DISPLAY_W//2 - draw_w, DISPLAY_H//2 - draw_h), 
                                           (DISPLAY_W//2 + draw_w, DISPLAY_H//2 + draw_h), (0, 255, 0), 2)
                cv2.putText(rgb_resized, f"RGB Feed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                latest_rgb_display = rgb_resized

            # --- 2. Process Deep Learning Depth & Calculate Volume ---
            if inLeft is not None and inRight is not None:
                depth_frame_count += 1 
                total_frames += 1
                
                frameLeft = inLeft.getCvFrame()
                frameRight = inRight.getCvFrame()

                # Format for HITNET and downscale for FPS
                left_bgr = cv2.cvtColor(frameLeft, cv2.COLOR_GRAY2BGR)
                right_bgr = cv2.cvtColor(frameRight, cv2.COLOR_GRAY2BGR)
                
                infer_w, infer_h = 320, 200
                left_infer = cv2.resize(left_bgr, (infer_w, infer_h))
                right_infer = cv2.resize(right_bgr, (infer_w, infer_h))

                # Run Hitnet Inference (Outputs Disparity)
                disparity_map = hitnet_depth(left_infer, right_infer)
                
                # --- CONVERT DISPARITY TO DEPTH (mm) ---
                # Avoid division by zero with + 1e-6
                depth_map_mm_lowres = (FOCAL_LENGTH_PIXELS * OAK_BASELINE_MM) / (disparity_map + 1e-6)
                
                # Scale the depth map back up to 640x400 for physical layout matching
                # Use INTER_NEAREST to avoid interpolating strict depth edges
                depth_map_mm = cv2.resize(depth_map_mm_lowres, (640, 400), interpolation=cv2.INTER_NEAREST)

                # Extract ROI Matrix
                height, width = depth_map_mm.shape
                cy, cx = height // 2, width // 2
                roi = depth_map_mm[cy - raw_half_box : cy + raw_half_box, 
                                   cx - raw_half_box : cx + raw_half_box]

                # --- VOLUME INTEGRATION MATH ---
                valid_pixels = (roi > 0) & (roi < (BASELINE_MM - HEIGHT_THRESHOLD))
                
                heights = np.zeros_like(roi, dtype=np.float32)
                heights[valid_pixels] = BASELINE_MM - roi[valid_pixels]
                
                volume_mm3 = np.sum(heights) * PIXEL_AREA_MM2
                volume_cm3 = volume_mm3 / 1000.0 # Convert to cubic centimeters
                
                # Create Visualization (Mapping mm limits to 0-255 colormap)
                depth_visual = np.clip(depth_map_mm, 0, MAX_DISTANCE_MM) 
                depth_visual = (depth_visual * (255.0 / MAX_DISTANCE_MM)).astype(np.uint8)
                depth_visual = 255 - depth_visual # Invert so closer is hotter (red)
                depth_visual[depth_map_mm == 0] = 0 
                depth_mapped = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

                depth_resized = cv2.resize(depth_mapped, (DISPLAY_W, DISPLAY_H))
                scale_x, scale_y = DISPLAY_W / width, DISPLAY_H / height
                draw_w, draw_h = int(raw_half_box * scale_x), int(raw_half_box * scale_y)

                # Draw UI Elements
                cv2.rectangle(depth_resized, (DISPLAY_W//2 - draw_w, DISPLAY_H//2 - draw_h), 
                                             (DISPLAY_W//2 + draw_w, DISPLAY_H//2 + draw_h), (255, 255, 255), 2)
                cv2.putText(depth_resized, f"HITNET Depth | FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(depth_resized, f"Volume: {volume_cm3:.2f} cm^3", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                
                depth_resized = draw_detailed_legend(depth_resized, MAX_DISTANCE_MM, step_size=500)
                latest_depth_display = depth_resized

                # --- Update 3D Matplotlib ---
                # if total_frames % 15 == 0:
                #     ax.clear() 
                #     roi_clipped = np.clip(roi, 0, BASELINE_MM + 500)
                #     roi_inverted = (BASELINE_MM + 500) - roi_clipped 
                #     ax.set_zlim(0, BASELINE_MM + 500) 
                #     ax.set_axis_off() 
                #     ax.set_title(f'Live 3D Profile (Vol: {volume_cm3:.0f} cm3)')
                #     ax.plot_surface(X, Y, roi_inverted, cmap='viridis', edgecolor='none')
                #     plt.pause(0.001)

            # --- 3. Combine and Show ---
            combined_frame = np.hstack((latest_rgb_display, latest_depth_display))
            cv2.imshow(window_name, combined_frame)

            # --- Key Press Logic ---
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('t'): # TARE FUNCTION
                valid_roi_pixels = roi[roi > 0]
                if len(valid_roi_pixels) > 0:
                    BASELINE_MM = float(np.median(valid_roi_pixels))
                    print(f"\n[TARE] New Baseline Set to: {BASELINE_MM:.1f} mm")
                else:
                    print("\n[TARE] Failed: No valid depth data in ROI.")
                    
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping and cleaning up...")
        plt.ioff()
        plt.close()
        cv2.destroyAllWindows()