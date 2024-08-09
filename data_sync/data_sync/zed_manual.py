#!/usr/bin/env/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from collections import deque
import time

class zed_manual_sync(Node):

    def __init__(self):
        super().__init__('zed_manual_sync')

        self.A_sub = self.create_subscription(CameraInfo, '/zed_left/zed_node_0/left/A', self.A_callback, 10)
        self.B_sub = self.create_subscription(Image, '/zed_left/zed_node_0/left/B', self.B_callback, 10)
        self.C_sub = self.create_subscription(CameraInfo, '/zed_right/zed_node_1/left/C', self.C_callback, 10)
        self.D_sub = self.create_subscription(Image, '/zed_right/zed_node_1/left/D', self.D_callback, 10)

        self.A_pub = self.create_publisher(CameraInfo, 'Left_0_info', 10)
        self.B_pub = self.create_publisher(Image, 'Left_0_image', 10)
        self.C_pub = self.create_publisher(CameraInfo, 'Right_1_info', 10)
        self.D_pub = self.create_publisher(Image, 'Right_1_image', 10)

        #queues to store incoming messages
        self.A_queue = deque()
        self.B_queue = deque()
        self.C_queue = deque()
        self.D_queue = deque()

        self.slop = 0.1

        self.get_logger().info("Zed Manual Sync Node has been started")

    def A_callback(self, msg):
        self.A_queue.append(msg)
        self.try_sync()

    def B_callback(self, msg):
        self.B_queue.append(msg)
        self.try_sync()

    def C_callback(self, msg):
        self.C_queue.append(msg)
        self.try_sync()

    def D_callback(self, msg):
        self.D_queue.append(msg)
        self.try_sync()

    def get_msg_time(self, msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def try_sync(self):

        if len(self.A_queue) > 0 and len(self.B_queue) > 0 and len(self.C_queue) > 0 and len(self.D_queue) > 0: #to ensure that the queues are not empty
            #get the timestamps of the messages at the leftmost of the queues (that is the oldest message in the queue)
            A_time = self.get_msg_time(self.A_queue[0])
            B_time = self.get_msg_time(self.B_queue[0])
            C_time = self.get_msg_time(self.C_queue[0])
            D_time = self.get_msg_time(self.D_queue[0])

            #calculate the max and min timestamps
            timestamps = [A_time, B_time, C_time, D_time]
            max_time = max(timestamps)
            min_time = min(timestamps)

            #check if the messages are synced
            if max_time - min_time <= self.slop:
                # If they are synchronized, publish them
                self.publish_synced_messages()
            else:
                # If they are not synchronized, discard the oldest message from the queue with the earliest timestamp
                self.discard_oldest_message()

    

    def publish_synced_messages(self):
        
        A_info = self.A_queue.popleft()
        B_image = self.B_queue.popleft()
        C_info = self.C_queue.popleft()
        D_image = self.D_queue.popleft()

        self.A_pub.publish(A_info)
        self.B_pub.publish(B_image)
        self.C_pub.publish(C_info)
        self.D_pub.publish(D_image)

    
    def discard_oldest_message(self):
        # Discard the message with the oldest timestamp
        queues = [self.A_queue, self.B_queue, self.C_queue, self.D_queue]
        earliest_time = min(self.get_msg_time(queue[0]) for queue in queues if queue)
        for queue in queues:
            if queue and self.get_msg_time(queue[0]) == earliest_time:
                queue.popleft()
                break


def main(args=None):
    rclpy.init(args=args)
    node = zed_manual_sync()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected...shutting the node down")
    
    rclpy.shutdown()


