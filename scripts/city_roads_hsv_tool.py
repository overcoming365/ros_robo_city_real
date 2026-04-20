#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class HsvTool(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.window = "city_roads_hsv_tool"
        self.image = None
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.create_trackbars()
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)

    def create_trackbars(self):
        config = [
            ("LH", 0, 180),
            ("LS", 0, 255),
            ("LV", 0, 255),
            ("UH", 180, 180),
            ("US", 255, 255),
            ("UV", 85, 255),
        ]
        for name, value, maximum in config:
            cv2.createTrackbar(name, self.window, value, maximum, lambda value: None)

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.image is None:
                rate.sleep()
                continue
            frame = self.image.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = (
                cv2.getTrackbarPos("LH", self.window),
                cv2.getTrackbarPos("LS", self.window),
                cv2.getTrackbarPos("LV", self.window),
            )
            upper = (
                cv2.getTrackbarPos("UH", self.window),
                cv2.getTrackbarPos("US", self.window),
                cv2.getTrackbarPos("UV", self.window),
            )
            mask = cv2.inRange(hsv, lower, upper)
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow(self.window, masked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("p"):
                print("line_hsv_lower: [%d, %d, %d]" % lower)
                print("line_hsv_upper: [%d, %d, %d]" % upper)
            if key == ord("q"):
                break
            rate.sleep()
        cv2.destroyAllWindows()


def main():
    rospy.init_node("city_roads_hsv_tool")
    HsvTool().run()


if __name__ == "__main__":
    main()
