import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import time
import os

class Kinect:

    def __init__(self,device_id=0):
        self.device_id=device_id
        self.k4a1_done_event = False
        self.k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_720P, depth_mode=pyk4a.DepthMode.NFOV_UNBINNED, camera_fps=pyk4a.FPS.FPS_30), device_id=self.device_id)
        self.k4a.start()

    def framerate(self):
        print("\nFramerate calculating ..")
        collection_duration = 15
        start_time = time.time()
        frame_count = 1

        try:            
            while time.time() - start_time < collection_duration:
                capture = self.k4a.get_capture()
                frame_count += 1
        except KeyboardInterrupt:
            print("Kinect-1 Interrupted")
        finally:
            print("kinect-1 Framerate:", (frame_count-1)/(time.time() - start_time))
            self.k4a.stop()


    def image_stream(self,path='./kinect'):

        color_directory = path + f"{self.device_id+1}_data/kinect_color/"
        depth_directory = path + f"{self.device_id+1}_data/kinect_depth/"
        os.makedirs(color_directory, exist_ok=True)
        os.makedirs(depth_directory, exist_ok=True)

        start_time = time.time()
        frame_count = 1

        try:
            while True:
                capture = self.k4a.get_capture()

                color_filename= os.path.join(color_directory, f'color_frame_{frame_count}.png')
                depth_filename = os.path.join(depth_directory, f'depth_frame_{frame_count}.png')

                cv2.imwrite(color_filename, cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR))
                cv2.imwrite(depth_filename, capture.transformed_depth)

                frame_count += 1
                print("Time:",round((time.time() - start_time),2), f"kinect-{self.device_id+1} frames:",frame_count)


        except KeyboardInterrupt:
            print(f"Kinect-{self.device_id+1} Record enterrrupted")
            
        finally:
            self.k4a.stop()
            print(f"kinect-{self.device_id+1} Framerate:", (frame_count-1)/(time.time() - start_time))


    def video_stream(self):

        print("press q to stop the stream.")

        try:
            while True:
                capture = self.k4a.get_capture()
                cv2.imshow(f'Kinect{self.device_id+1}Feed', capture.color)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Feed stopped by user.")
            
        finally:
            self.k4a.stop()
            cv2.destroyAllWindows()


    def capture_single_frame(self,path='./kinect', frame_id=None):

        color_directory = path + f"{self.device_id+1}_data/kinect_color/"
        depth_directory = path + f"{self.device_id+1}_data/kinect_depth/"
        os.makedirs(color_directory, exist_ok=True)
        os.makedirs(depth_directory, exist_ok=True)

        frame_id = frame_id

        try:
            capture = self.k4a.get_capture()

            color_filename= os.path.join(color_directory, f'color_frame_{frame_id}.png')
            depth_filename= os.path.join(depth_directory, f'depth_frame_{frame_id}.png')

            cv2.imwrite(color_filename, cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR))
            cv2.imwrite(depth_filename, capture.transformed_depth)

        except KeyboardInterrupt:
            print(f"Kinect-{self.device_id+1} Record enterrrupted")
            
        finally:
            pass


if __name__ == "__main__":

    kinect1 = Kinect(device_id=0)
    # kinect2 = Kinect(device_id=1)

    # kinect1.framerate()
    # kinect2.framerate()
    # kinect.image_stream()
    # kinect1.video_stream()
    # kinect2.video_stream()

    start_time = time.time()
    total_frame=900

    for i in range(total_frame):
        print(i)
        kinect1.capture_single_frame(path='./kinect',frame_id=i)
        # time.sleep(0.1)

    frame_rate = (total_frame-1)/(time.time() - start_time)
    print("kinect framerate:", frame_rate)

    # kinect1.capture_single_frame(30)

