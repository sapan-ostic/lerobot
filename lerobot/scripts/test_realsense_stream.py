#!/usr/bin/env python3
"""
Intel RealSense Camera Live Video Test Script

This script connects to your Intel RealSense camera and displays live video.
It also shows camera information and allows you to save frames.

Usage:
    python test_realsense_camera.py

Controls:
    - Press 'q' to quit
    - Press 's' to save current frame
    - Press 'd' to toggle depth view (if available)
    - Press 'i' to show camera info
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

class RealSenseViewer:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.depth_enabled = False
        self.show_depth = False
        self.frame_count = 0
        
    def initialize_camera(self):
        """Initialize the RealSense camera"""
        try:
            # Create a context object. This object owns the handles to all connected realsense devices
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                print("ERROR: No RealSense devices found!")
                return False
                
            # Get device info
            device = devices[0]
            print(f"Found RealSense device: {device.get_info(rs.camera_info.name)}")
            
            try:
                serial = device.get_info(rs.camera_info.serial_number)
                print(f"Serial number: {serial}")
            except:
                print("Could not get serial number")
                
            try:
                product_id = device.get_info(rs.camera_info.product_id)
                print(f"Product ID: {product_id}")
            except:
                print("Could not get product ID")
            
            # Create a pipeline
            self.pipeline = rs.pipeline()
            
            # Create a config and configure streams
            self.config = rs.config()
            
            # Configure color stream
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Try to enable depth stream
            try:
                self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                self.depth_enabled = True
                print("Depth stream enabled")
            except Exception as e:
                print(f"Depth stream not available: {e}")
                self.depth_enabled = False
            
            # Start streaming
            print("Starting camera stream...")
            self.pipeline.start(self.config)
            print("Camera initialized successfully!")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure your RealSense camera is connected")
            print("2. Check USB permissions:")
            print("   sudo chmod 666 /dev/bus/usb/002/003  # Adjust bus/device numbers")
            print("3. Try reconnecting the camera")
            return False
    
    def get_frames(self):
        """Get frames from the camera"""
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if self.depth_enabled else None
            
            if not color_frame:
                return None, None
                
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None
    
    def save_frame(self, color_image, depth_image=None):
        """Save current frame to disk"""
        timestamp = int(time.time())
        
        # Create output directory
        os.makedirs("realsense_captures", exist_ok=True)
        
        # Save color image
        color_filename = f"realsense_captures/color_{timestamp}_{self.frame_count:04d}.jpg"
        cv2.imwrite(color_filename, color_image)
        print(f"Saved color image: {color_filename}")
        
        # Save depth image if available
        if depth_image is not None:
            depth_filename = f"realsense_captures/depth_{timestamp}_{self.frame_count:04d}.png"
            cv2.imwrite(depth_filename, depth_image)
            print(f"Saved depth image: {depth_filename}")
    
    def run(self):
        """Main loop to display live video"""
        if not self.initialize_camera():
            return
            
        print("\nLive video started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        if self.depth_enabled:
            print("  'd' - Toggle depth view")
        print("  'i' - Show camera info")
        print()
        
        try:
            while True:
                color_image, depth_image = self.get_frames()
                
                if color_image is None:
                    continue
                
                self.frame_count += 1
                
                # Choose which image to display
                if self.show_depth and depth_image is not None:
                    # Apply colormap to depth image for better visualization
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    display_image = depth_colormap
                    window_title = "RealSense - Depth View"
                else:
                    display_image = color_image
                    window_title = "RealSense - Color View"
                
                # Add frame counter to image
                cv2.putText(display_image, f"Frame: {self.frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show FPS
                cv2.putText(display_image, f"30 FPS", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the image
                cv2.imshow(window_title, display_image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    self.save_frame(color_image, depth_image)
                elif key == ord('d') and self.depth_enabled:
                    self.show_depth = not self.show_depth
                    print(f"Switched to {'depth' if self.show_depth else 'color'} view")
                elif key == ord('i'):
                    print(f"\nCamera Info:")
                    print(f"  Frame count: {self.frame_count}")
                    print(f"  Resolution: {color_image.shape[1]}x{color_image.shape[0]}")
                    print(f"  Depth enabled: {self.depth_enabled}")
                    print(f"  Current view: {'depth' if self.show_depth else 'color'}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.pipeline:
                self.pipeline.stop()
                print("Camera pipeline stopped")
        except:
            pass
        
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    print("Intel RealSense Camera Test")
    print("=" * 40)
    
    viewer = RealSenseViewer()
    viewer.run()

if __name__ == "__main__":
    main()
