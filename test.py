import os
import open3d as o3d
import numpy as np
import cv2
import PIL.Image as Image
from matplotlib import pyplot as plt
from transformers import  SamModel, SamProcessor
import torch

# Import the classes from your script
from segment_point_cloud import SamPointCloudSegmenter
from grounded_sam import GroundedSAM 

class CloudSegmentor:
    def __init__(self):
        # Define the directory path where your files are stored
        self.directory_path = "/home/stretch/Documents/bree/3D_perception/point_cloud/sam3d/"  # Replace with your actual directory path

        # List files in the directory
        self.files = os.listdir(self.directory_path)

    def load_base_supp(self, base_image_path, base_cloud_path, supp_image_path, supp_cloud_path):
        # Define the paths for your base and supplementary pairs of 2D image and 3D point cloud
        base_rgb_image_file = os.path.join(self.directory_path,base_image_path)
        base_point_cloud_file = os.path.join(base_cloud_path)
        supplementary_rgb_image_file = os.path.join(supp_image_path)
        supplementary_point_cloud_file = os.path.join(supp_image_path)

        # Assuming one base image-point cloud pair and one supplementary image-point cloud pair
        for file in self.files:
            if file.endswith(".jpg") or file.endswith(".png"):
                if not base_rgb_image_file:
                    base_rgb_image_file = os.path.join(self.directory_path, file)
                else:
                    supplementary_rgb_image_file = os.path.join(self.directory_path, file)
            elif file.endswith(".ply") or file.endswith(".pcd"):
                if not base_point_cloud_file:
                    base_point_cloud_file = os.path.join(self.directory_path, file)
                else:
                    supplementary_point_cloud_file = os.path.join(self.directory_path, file)

        # Ensure both the base and supplementary files are found
        if not base_rgb_image_file or not base_point_cloud_file or not supplementary_rgb_image_file or not supplementary_point_cloud_file:
            raise ValueError("Could not find the required base and supplementary image-point cloud pairs in the directory.")

        # Load the base RGB image
        base_rgb_image = cv2.imread(base_rgb_image_file)
        base_rgb_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2RGB)
        base_rgb_image = cv2.resize(base_rgb_image, (1024, 1024))  # Resize for consistency
        self.base_rgb_image_pil = Image.fromarray(base_rgb_image)

        # Load the base point cloud (either .ply or .pcd)
        base_pcd = o3d.io.read_point_cloud(base_point_cloud_file)
        base_point_cloud_data = np.asarray(base_pcd.points)

        # Load the supplementary RGB image
        supplementary_rgb_image = cv2.imread(supplementary_rgb_image_file)
        supplementary_rgb_image = cv2.cvtColor(supplementary_rgb_image, cv2.COLOR_BGR2RGB)
        supplementary_rgb_image = cv2.resize(supplementary_rgb_image, (1024, 1024))  # Resize for consistency
        self.supplementary_rgb_image_pil = Image.fromarray(supplementary_rgb_image)

        # Load the supplementary point cloud (either .ply or .pcd)
        supplementary_pcd = o3d.io.read_point_cloud(supplementary_point_cloud_file)
        self.supplementary_point_cloud_data = np.asarray(supplementary_pcd.points)
    
    def segment_cloud(self, bounding_box):
        # Example of a bounding box (adjust this as per your requirement)
        bounding_box = [300, 300, 700, 700]  # Example coordinates of the bounding box [xmin, ymin, xmin, ymax]

        # Create the SamPointCloudSegmenter instance
        segmenter = SamPointCloudSegmenter(device='cpu', render_2d_results=True)

        # Call the segment method
        self.segmented_points, self.segmented_colors, self.segmentation_masks = segmenter.segment(
            self.base_rgb_image_pil, self.base_point_cloud_data, bounding_box, 
            [self.supplementary_rgb_image_pil], [self.supplementary_point_cloud_data]
        )
        print("Segmented point", self.segmented_points)
        print("Segmented colors", self.segmented_colors)
        print("Segmented point", self.segmentation_masks)

    def show_cloud_seg(self):
        # Visualize the segmented point cloud
        segmented_pcd = o3d.geometry.PointCloud()
        segmented_pcd.points = o3d.utility.Vector3dVector(self.segmented_points)
        segmented_pcd.colors = o3d.utility.Vector3dVector(self.segmented_colors / 255.0)  # Normalize colors

        # Visualize with Open3D
        o3d.visualization.draw_geometries([segmented_pcd.points])

        # Optionally, visualize the segmentation mask (2D)
        for points in self.segmentation_masks:
            plt.imshow(points[0, 0].detach().cpu().numpy(), alpha=0.5)
            plt.title("Segmentation Mask")
            plt.show()

    def get_cloud(depth_image):
        # TODO: covert

if __name__ == "__main__":
    cloudSeg = CloudSegmentor()
    cloudSeg.load_base_supp("RobotPOV.jpg", "robotpov.pcd", "KinectPOV.jpg", "kinectpov.pcd")

    gsam = GroundedSAM()
    gsam.load_image("/home/stretch/Documents/bree/3D_perception/point_cloud/sam3d/RobotPOV.jpg") 
    boxes = gsam.get_detections() # get base image bounding box with no query
    cloudSeg.segment_cloud(boxes[0]) # just use one box for now
    cloudSeg.show_cloud_seg()

    # TODO: get point cloud from kinect depth image!! should be a library for converting 
    """
    https://github.com/HarendraKumarSingh/stereo-images-to-3D-model-generation/blob/master/depth-map-to-3D-point-cloud.ipynb
    https://amanjaglan.medium.com/generating-3d-images-and-point-clouds-with-python-open3d-and-glpn-for-depth-estimation-a0a484d77570
    """