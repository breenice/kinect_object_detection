import cv2
import numpy as np
import time
import os
import pyrealsense2 as rs

class RealSense:
    def __init__(self):

        self.color_image = None
        self.depth_image = None
        self.image_width= 1280
        self.image_height= 720
        self.fps=30
        self.frames= None

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.image_width, self.image_height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.image_width,self.image_height, rs.format.z16, self.fps)
        self.pipeline.start(config)

    def framerate(self):
        start_time = time.time()
        frame_count = 1
        collection_duration=20
        print("\nFramerate calculating ..")

        try:
             while time.time() - start_time < collection_duration:

                self.frames = self.pipeline.wait_for_frames()
                color_frame = self.frames.get_color_frame()
                depth_frame = self.frames.get_depth_frame()
                self.color_image = np.asanyarray(color_frame.get_data())
                self.depth_image = np.asanyarray(depth_frame.get_data())

                frame_count +=1

        finally:
            frame_rate = (frame_count-1)/(time.time() - start_time)
            print("relsense framerate:", frame_rate)

    def image_stream(self,path='./rs_data'):

        self.color_output_directory = path + "/real_color/"
        self.depth_output_directory = path + "/real_depth/"

        os.makedirs(self.color_output_directory, exist_ok=True)
        os.makedirs(self.depth_output_directory, exist_ok=True)

        start_time = time.time()
        frame_count = 1

        try:
             while True:

                self.frames = self.pipeline.wait_for_frames()
                color_frame = self.frames.get_color_frame()
                depth_frame = self.frames.get_depth_frame()
                self.color_image = np.asanyarray(color_frame.get_data())
                self.depth_image = np.asanyarray(depth_frame.get_data())

                color_filename = os.path.join(self.color_output_directory, f"color_{frame_count}_{time.time()}.png")
                depth_filename = os.path.join(self.depth_output_directory, f"depth_{frame_count}_{time.time()}.png")
                cv2.imwrite(color_filename, self.color_image)
                cv2.imwrite(depth_filename, self.depth_image)

                frame_count +=1
                print("frame_count",frame_count)

        except KeyboardInterrupt:
            print("Stopped by user.")

        finally:
           print("Realsense Framerate:", (frame_count-1)/(time.time() - start_time))
           self.pipeline.stop()

    def video_stream(self):

        print("press q to stop the stream.")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow('RealSense Feed', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Feed stopped by user.")

        finally:
           self.pipeline.stop()
           cv2.destroyAllWindows()

    def write_single_frame(self, path='./rs_data', frame_id=None):

        self.color_output_directory = path + "/real_color/"
        self.depth_output_directory = path + "/real_depth/"

        os.makedirs(self.color_output_directory, exist_ok=True)
        os.makedirs(self.depth_output_directory, exist_ok=True)

        frame_id = frame_id

        try:
             
            self.frames = self.pipeline.wait_for_frames()
            color_frame = self.frames.get_color_frame()
            depth_frame = self.frames.get_depth_frame()
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())

            color_filename = os.path.join(self.color_output_directory, f"color_{frame_id}_{time.time()}.png")
            depth_filename = os.path.join(self.depth_output_directory, f"depth_{frame_id}_{time.time()}.png")

            cv2.imwrite(color_filename, self.color_image)
            cv2.imwrite(depth_filename, self.depth_image)

        except KeyboardInterrupt:
            print("Stopped by user.")

        finally:
           pass

    def get_single_rgbd_frame(self):

        try:
             
            self.frames = self.pipeline.wait_for_frames()
            color_frame = self.frames.get_color_frame()
            depth_frame = self.frames.get_depth_frame()
            # self.color_image = np.asanyarray(color_frame.get_data())
            # self.depth_image = np.asanyarray(depth_frame.get_data())

            return color_frame, depth_frame

        except KeyboardInterrupt:
            print("Stopped by user.")

        finally:
           pass

    def stop_realsense(self):
           self.pipeline.stop()

if __name__ == "__main__":
    realsense = RealSense()
    # realsense.framerate()
    realsense.video_stream()
    # realsense.image_stream()


###################################################
    
    # rs_color_frames = []
    # rs_depth_frames = []
    # timestamps = []

    # for i in range(1000):
    #     print("i:",i)

    #     color_frame, depth_frame = realsense.get_single_rgbd_frame()
    #     rs_color_frames.append(color_frame)
    #     rs_depth_frames.append(depth_frame)
    #     timestamps.append(time.time())

    # path='./rs_data_test'
    # color_output_directory = path + "/real_color/"
    # depth_output_directory = path + "/real_depth/"
    # os.makedirs(color_output_directory, exist_ok=True)
    # os.makedirs(depth_output_directory, exist_ok=True)

    # for i, (color_frame, depth_frame, timestamp) in enumerate(zip(rs_color_frames, rs_depth_frames, timestamps)):
    #     color_image = np.asanyarray(color_frame.get_data())
    #     depth_image = np.asanyarray(depth_frame.get_data())

    #     color_filename = os.path.join(color_output_directory, f"color_frame_{i}_{timestamp}.png")
    #     depth_filename = os.path.join(depth_output_directory, f"depth_frame_{i}_{timestamp}.png")

    #     cv2.imwrite(color_filename, color_image)
    #     cv2.imwrite(depth_filename, depth_image)

    #     print(f"(Writing) Realsense frame {i}")

    # print("Realsense image writing done.")

###################################################
    
    # start_time = time.time()
    # total_frame=900

    # for i in range(total_frame):
    #     print("frane_id:",i)
    #     realsense.write_single_frame(frame_id=i)
    #     # time.sleep(1)

    # frame_rate = (total_frame-1)/(time.time() - start_time)
    # print("relsense framerate:", frame_rate)


