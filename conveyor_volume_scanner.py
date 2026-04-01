import depthai as dai
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# --- System & Math Constants ---
DISPLAY_W = 800
DISPLAY_H = 600
RAW_BOX_SIZE = 200
raw_half_box = RAW_BOX_SIZE // 2  
MAX_DISTANCE_MM = 3000 

# Volume Calculation Constants (You will need to calibrate these!)
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

# --- Main Setup ---
pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xout_depth = pipeline.create(dai.node.XLinkOut)

camRgb = pipeline.create(dai.node.ColorCamera)
xout_rgb = pipeline.create(dai.node.XLinkOut)

xout_depth.setStreamName("depth")
xout_rgb.setStreamName("rgb")

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(60)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(60)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setLeftRightCheck(True)

camRgb.setPreviewSize(640, 480) 
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xout_depth.input)
camRgb.preview.link(xout_rgb.input)

# --- Main Loop ---
print("Connecting to OAK-D Pro...")
with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(1200) 
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
    window_name = "OAK-D Conveyor Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W * 2, DISPLAY_H)

    startTime = time.monotonic()
    depth_frame_count = 0 
    total_frames = 0
    fps = 0

    latest_rgb_display = None
    latest_depth_display = None
    volume_cm3 = 0.0

    print("Dashboard and Integration started! Press 'q' to quit.")

    try:
        while True:
            inDepth = q_depth.tryGet()
            inRgb = q_rgb.tryGet()

            currentTime = time.monotonic()
            if (currentTime - startTime) > 1: 
                fps = depth_frame_count / (currentTime - startTime)
                depth_frame_count = 0
                startTime = currentTime

            # --- 1. Process Depth & Calculate Volume ---
            if inDepth is not None:
                depth_frame_count += 1 
                total_frames += 1
                depth_frame = inDepth.getFrame()
                
                # Extract ROI Matrix
                height, width = depth_frame.shape
                cy, cx = height // 2, width // 2
                roi = depth_frame[cy - raw_half_box : cy + raw_half_box, 
                                  cx - raw_half_box : cx + raw_half_box]

                # --- VOLUME INTEGRATION MATH ---
                # 1. Ignore noise: pixels must be > 0 and closer than the baseline
                valid_pixels = (roi > 0) & (roi < (BASELINE_MM - HEIGHT_THRESHOLD))
                
                # --- VISUALIZE THE MASK ---
                # Convert the True/False mask to a Black/White image (0 or 255)
                mask_visual = (valid_pixels * 255).astype(np.uint8)
                # Resize it so it's easier to see on your screen
                mask_visual = cv2.resize(mask_visual, (400, 400), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Valid Pixels Mask (White = Load)", mask_visual)
                # --------------------------
                
                # 2. Create an empty matrix for the heights
                heights = np.zeros_like(roi, dtype=np.float32)
                
                # 3. Calculate height (Baseline - Measured) for valid pixels only
                heights[valid_pixels] = BASELINE_MM - roi[valid_pixels]
                
                # 4. Integrate (Sum) and multiply by pixel area
                volume_mm3 = np.sum(heights) * PIXEL_AREA_MM2
                volume_cm3 = volume_mm3 / 1000.0 # Convert to cubic centimeters
                # -------------------------------
                
                # Create Visualization
                depth_visual = np.clip(depth_frame, 0, MAX_DISTANCE_MM) 
                depth_visual = (depth_visual * (255.0 / MAX_DISTANCE_MM)).astype(np.uint8)
                depth_visual = 255 - depth_visual
                depth_visual[depth_frame == 0] = 0 
                depth_mapped = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

                depth_resized = cv2.resize(depth_mapped, (DISPLAY_W, DISPLAY_H))
                scale_x, scale_y = DISPLAY_W / width, DISPLAY_H / height
                draw_w, draw_h = int(raw_half_box * scale_x), int(raw_half_box * scale_y)

                # Draw UI Elements
                cv2.rectangle(depth_resized, (DISPLAY_W//2 - draw_w, DISPLAY_H//2 - draw_h), 
                                             (DISPLAY_W//2 + draw_w, DISPLAY_H//2 + draw_h), (255, 255, 255), 2)
                cv2.putText(depth_resized, f"Depth Map | FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display the Live Volume!
                cv2.putText(depth_resized, f"Volume: {volume_cm3:.2f} cm^3", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                
                depth_resized = draw_detailed_legend(depth_resized, MAX_DISTANCE_MM, step_size=500)
                latest_depth_display = depth_resized

                # --- Update 3D Matplotlib (Throttled to every 15 frames) ---
                if total_frames % 15 == 0:
                    ax.clear() 
                    roi_clipped = np.clip(roi, 0, BASELINE_MM + 500)
                    roi_inverted = (BASELINE_MM + 500) - roi_clipped 
                    ax.set_zlim(0, BASELINE_MM + 500) 
                    ax.set_axis_off() 
                    ax.set_title(f'Live 3D Profile (Vol: {volume_cm3:.0f} cm3)')
                    surf = ax.plot_surface(X, Y, roi_inverted, cmap='viridis', edgecolor='none')
                    plt.pause(0.001)

            # --- 2. Process RGB ---
            if inRgb is not None:
                rgb_frame = inRgb.getCvFrame() 
                rgb_resized = cv2.resize(rgb_frame, (DISPLAY_W, DISPLAY_H))
                scale_x, scale_y = DISPLAY_W / rgb_frame.shape[1], DISPLAY_H / rgb_frame.shape[0]
                draw_w, draw_h = int(raw_half_box * scale_x), int(raw_half_box * scale_y)
                
                cv2.rectangle(rgb_resized, (DISPLAY_W//2 - draw_w, DISPLAY_H//2 - draw_h), 
                                           (DISPLAY_W//2 + draw_w, DISPLAY_H//2 + draw_h), (0, 255, 0), 2)
                cv2.putText(rgb_resized, f"RGB Feed | FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                latest_rgb_display = rgb_resized

            # --- 3. Combine and Show ---
            if latest_rgb_display is not None and latest_depth_display is not None:
                combined_frame = np.hstack((latest_rgb_display, latest_depth_display))
                cv2.imshow(window_name, combined_frame)

            # --- Key Press Logic ---
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('t'): # TARE FUNCTION
                # When 't' is pressed, set the baseline to the median depth of the current ROI
                # (We use median to ignore random noise spikes)
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