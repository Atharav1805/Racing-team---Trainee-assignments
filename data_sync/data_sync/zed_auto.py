#!/usr/bin/env/python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber

class zed_auto_sync(Node):

    def __init__(self):

        super().__init__('zed_auto_sync') #create a node called zed_manual_sync

        self.A = Subscriber(self, CameraInfo, '/zed_left/zed_node_0/left/A')
        self.B = Subscriber(self, Image, '/zed_left/zed_node_0/left/B')
        self.C = Subscriber(self, CameraInfo, '/zed_right/zed_node_1/left/C')
        self.D = Subscriber(self, Image, '/zed_right/zed_node_1/left/D')

        self.sync = ApproximateTimeSynchronizer([self.A, self.B, self.C, self.D], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.sync_callback)

        self.A_pub = self.create_publisher(CameraInfo, 'Left_0_info', 10) 
        self.B_pub = self.create_publisher(Image, 'Left_0_image', 10)
        self.C_pub = self.create_publisher(CameraInfo, 'Right_1_info', 10)
        self.D_pub = self.create_publisher(Image, 'Right_1_image', 10)

        self.get_logger().info("Zed Auto Sync Node has been started")

    def sync_callback(self, A_info, B_image, C_info, D_image):

        self.A_pub.publish(A_info)
        self.B_pub.publish(B_image)
        self.C_pub.publish(C_info)
        self.D_pub.publish(D_image)
        
        

def main(args=None):

    rclpy.init(args=args)

    node = zed_auto_sync()

    try: 

        rclpy.spin(node)
    
    except KeyboardInterrupt:

        node.get_logger().info("Keyboard Interrupt detected...shutting the node down")
        pass
    
    rclpy.shutdown()