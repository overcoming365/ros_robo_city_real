#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import math
import os
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def now_sec():
    return time.time()


def ratio_to_rect(width, height, ratio_rect):
    x1 = int(width * ratio_rect[0])
    y1 = int(height * ratio_rect[1])
    x2 = int(width * ratio_rect[2])
    y2 = int(height * ratio_rect[3])
    return x1, y1, x2, y2


def hsv_mask(image_hsv, lower, upper):
    lower_np = np.array(lower, dtype=np.uint8)
    upper_np = np.array(upper, dtype=np.uint8)
    return cv2.inRange(image_hsv, lower_np, upper_np)


def find_contours(binary_image):
    result = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 3:
        return result[1]
    return result[0]


class TemplateMatcher(object):
    def __init__(self, template_dir, file_map, scales, threshold):
        self.templates = {}
        self.scales = scales
        self.threshold = threshold
        for label, filename in file_map.items():
            path = filename
            if not os.path.isabs(path):
                path = os.path.join(template_dir, filename)
            if not os.path.exists(path):
                continue
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            self.templates[label] = cv2.Canny(image, 50, 150)

    def detect(self, frame_bgr, ratio_rect):
        if not self.templates:
            return None, 0.0, None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = ratio_to_rect(w, h, ratio_rect)
        roi_bgr = frame_bgr[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return None, 0.0, None
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, 50, 150)

        best_label = None
        best_score = self.threshold
        best_box = None

        for label, tpl in self.templates.items():
            for scale in self.scales:
                resized = cv2.resize(tpl, None, fx=scale, fy=scale)
                th, tw = resized.shape[:2]
                if th < 12 or tw < 12:
                    continue
                if roi_edges.shape[0] < th or roi_edges.shape[1] < tw:
                    continue
                res = cv2.matchTemplate(roi_edges, resized, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score = score
                    best_label = label
                    best_box = (x1 + loc[0], y1 + loc[1], tw, th)
        return best_label, best_score, best_box


class CityRoadsController(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.frame_index = 0
        self.last_line_error = 0.0
        self.last_voice = {}
        self.front_min_range = float("inf")

        self.speed_mode = "cruise"
        self.pending_stop_reason = None
        self.last_crosswalk_time = 0.0
        self.last_cone_time = 0.0
        self.danger_seen_time = None
        self.finish_done = False

        self.active_mode = None
        self.mode_until = 0.0
        self.mode_angular = 0.0
        self.mode_linear = 0.0

        self.team_name = rospy.get_param("~team_name", rospy.get_param("/team_name", "TeamName"))
        self.image_topic = rospy.get_param("~image_topic", rospy.get_param("/image_topic", "/usb_cam/image_raw"))
        self.scan_topic = rospy.get_param("~scan_topic", rospy.get_param("/scan_topic", "/scan"))
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", rospy.get_param("/cmd_vel_topic", "/cmd_vel"))
        self.debug_image_topic = rospy.get_param("~debug_image_topic", rospy.get_param("/debug_image_topic", "/city_roads/debug_image"))
        self.voice_topic = rospy.get_param("~voice_topic", rospy.get_param("/voice_topic", "/city_roads/voice"))
        self.frame_skip = int(rospy.get_param("~frame_skip", rospy.get_param("/frame_skip", 1)))
        self.use_laser = bool(rospy.get_param("~use_laser", rospy.get_param("/use_laser", True)))
        self.publish_debug_image = bool(rospy.get_param("~publish_debug_image", rospy.get_param("/publish_debug_image", True)))

        self.line_hsv_lower = rospy.get_param("~line_hsv_lower", rospy.get_param("/line_hsv_lower", [0, 0, 0]))
        self.line_hsv_upper = rospy.get_param("~line_hsv_upper", rospy.get_param("/line_hsv_upper", [180, 255, 85]))
        self.line_roi = rospy.get_param("~line_roi", rospy.get_param("/line_roi", [0.0, 0.58, 1.0, 1.0]))
        self.line_search_min_area = int(rospy.get_param("~line_search_min_area", rospy.get_param("/line_search_min_area", 250)))
        self.line_center_bias_px = int(rospy.get_param("~line_center_bias_px", rospy.get_param("/line_center_bias_px", 0)))
        self.line_kp = float(rospy.get_param("~line_kp", rospy.get_param("/line_kp", 0.0055)))
        self.line_kd = float(rospy.get_param("~line_kd", rospy.get_param("/line_kd", 0.0010)))
        self.line_search_turn_speed = float(rospy.get_param("~line_search_turn_speed", rospy.get_param("/line_search_turn_speed", 0.22)))

        self.cruise_speed = float(rospy.get_param("~cruise_speed", rospy.get_param("/cruise_speed", 0.20)))
        self.limited_speed = float(rospy.get_param("~limited_speed", rospy.get_param("/limited_speed", 0.10)))
        self.turn_speed = float(rospy.get_param("~turn_speed", rospy.get_param("/turn_speed", 0.12)))
        self.lane_change_speed = float(rospy.get_param("~lane_change_speed", rospy.get_param("/lane_change_speed", 0.12)))
        self.cone_avoid_speed = float(rospy.get_param("~cone_avoid_speed", rospy.get_param("/cone_avoid_speed", 0.10)))
        self.finish_speed = float(rospy.get_param("~finish_speed", rospy.get_param("/finish_speed", 0.10)))
        self.max_angular_speed = float(rospy.get_param("~max_angular_speed", rospy.get_param("/max_angular_speed", 0.55)))

        self.crosswalk_white_threshold = int(rospy.get_param("~crosswalk_white_threshold", rospy.get_param("/crosswalk_white_threshold", 185)))
        self.crosswalk_roi = rospy.get_param("~crosswalk_roi", rospy.get_param("/crosswalk_roi", [0.15, 0.52, 0.85, 0.76]))
        self.crosswalk_min_row_ratio = float(rospy.get_param("~crosswalk_min_row_ratio", rospy.get_param("/crosswalk_min_row_ratio", 0.32)))
        self.crosswalk_min_groups = int(rospy.get_param("~crosswalk_min_groups", rospy.get_param("/crosswalk_min_groups", 4)))
        self.crosswalk_stop_seconds = float(rospy.get_param("~crosswalk_stop_seconds", rospy.get_param("/crosswalk_stop_seconds", 2.0)))
        self.crosswalk_cooldown_seconds = float(rospy.get_param("~crosswalk_cooldown_seconds", rospy.get_param("/crosswalk_cooldown_seconds", 5.0)))

        self.traffic_light_roi = rospy.get_param("~traffic_light_roi", rospy.get_param("/traffic_light_roi", [0.35, 0.05, 0.65, 0.28]))
        self.traffic_light_red_area = int(rospy.get_param("~traffic_light_red_area", rospy.get_param("/traffic_light_red_area", 120)))
        self.traffic_light_green_area = int(rospy.get_param("~traffic_light_green_area", rospy.get_param("/traffic_light_green_area", 120)))
        self.tl_red1_lower = rospy.get_param("~traffic_light_red1_lower", rospy.get_param("/traffic_light_red1_lower", [0, 100, 100]))
        self.tl_red1_upper = rospy.get_param("~traffic_light_red1_upper", rospy.get_param("/traffic_light_red1_upper", [12, 255, 255]))
        self.tl_red2_lower = rospy.get_param("~traffic_light_red2_lower", rospy.get_param("/traffic_light_red2_lower", [165, 100, 100]))
        self.tl_red2_upper = rospy.get_param("~traffic_light_red2_upper", rospy.get_param("/traffic_light_red2_upper", [180, 255, 255]))
        self.tl_green_lower = rospy.get_param("~traffic_light_green_lower", rospy.get_param("/traffic_light_green_lower", [40, 80, 80]))
        self.tl_green_upper = rospy.get_param("~traffic_light_green_upper", rospy.get_param("/traffic_light_green_upper", [95, 255, 255]))

        self.cone_roi = rospy.get_param("~cone_roi", rospy.get_param("/cone_roi", [0.20, 0.36, 0.80, 0.92]))
        self.cone_min_area = int(rospy.get_param("~cone_min_area", rospy.get_param("/cone_min_area", 700)))
        self.cone_red1_lower = rospy.get_param("~cone_red1_lower", rospy.get_param("/cone_red1_lower", [0, 120, 80]))
        self.cone_red1_upper = rospy.get_param("~cone_red1_upper", rospy.get_param("/cone_red1_upper", [12, 255, 255]))
        self.cone_red2_lower = rospy.get_param("~cone_red2_lower", rospy.get_param("/cone_red2_lower", [165, 120, 80]))
        self.cone_red2_upper = rospy.get_param("~cone_red2_upper", rospy.get_param("/cone_red2_upper", [180, 255, 255]))
        self.cone_avoid_seconds = float(rospy.get_param("~cone_avoid_seconds", rospy.get_param("/cone_avoid_seconds", 1.2)))
        self.cone_avoid_turn = float(rospy.get_param("~cone_avoid_turn", rospy.get_param("/cone_avoid_turn", 0.42)))
        self.cone_cooldown_seconds = float(rospy.get_param("~cone_cooldown_seconds", rospy.get_param("/cone_cooldown_seconds", 4.0)))

        self.laser_front_angle_deg = float(rospy.get_param("~laser_front_angle_deg", rospy.get_param("/laser_front_angle_deg", 16)))
        self.laser_front_stop_distance = float(rospy.get_param("~laser_front_stop_distance", rospy.get_param("/laser_front_stop_distance", 0.55)))

        self.finish_line_roi = rospy.get_param("~finish_line_roi", rospy.get_param("/finish_line_roi", [0.05, 0.78, 0.95, 0.98]))
        self.finish_line_white_threshold = int(rospy.get_param("~finish_line_white_threshold", rospy.get_param("/finish_line_white_threshold", 190)))
        self.finish_line_min_ratio = float(rospy.get_param("~finish_line_min_ratio", rospy.get_param("/finish_line_min_ratio", 0.40)))
        self.finish_roll_seconds = float(rospy.get_param("~finish_roll_seconds", rospy.get_param("/finish_roll_seconds", 0.8)))

        self.turn_seconds = float(rospy.get_param("~turn_seconds", rospy.get_param("/turn_seconds", 1.2)))
        self.turn_left_angular = float(rospy.get_param("~turn_left_angular", rospy.get_param("/turn_left_angular", 0.42)))
        self.turn_right_angular = float(rospy.get_param("~turn_right_angular", rospy.get_param("/turn_right_angular", -0.42)))
        self.straight_seconds = float(rospy.get_param("~straight_seconds", rospy.get_param("/straight_seconds", 0.5)))
        self.lane_change_default_direction = rospy.get_param("~lane_change_default_direction", rospy.get_param("/lane_change_default_direction", "left"))
        self.lane_change_seconds = float(rospy.get_param("~lane_change_seconds", rospy.get_param("/lane_change_seconds", 1.0)))
        self.lane_change_angular = float(rospy.get_param("~lane_change_angular", rospy.get_param("/lane_change_angular", 0.36)))
        self.voice_cooldown_seconds = float(rospy.get_param("~voice_cooldown_seconds", rospy.get_param("/voice_cooldown_seconds", 2.0)))

        template_dir = rospy.get_param("~template_dir", rospy.get_param("/template_dir", ""))
        if template_dir.startswith("$(find city_roads)"):
            template_dir = template_dir.replace("$(find city_roads)", os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        template_scales = rospy.get_param("~template_match_scales", rospy.get_param("/template_match_scales", [0.8, 1.0, 1.2]))
        template_threshold = float(rospy.get_param("~template_match_threshold", rospy.get_param("/template_match_threshold", 0.55)))
        self.template_detect_every_n_frames = int(rospy.get_param("~template_detect_every_n_frames", rospy.get_param("/template_detect_every_n_frames", 3)))
        self.template_roi = rospy.get_param("~template_roi", rospy.get_param("/template_roi", [0.10, 0.10, 0.90, 0.55]))
        sign_templates = rospy.get_param("~sign_templates", rospy.get_param("/sign_templates", {}))
        self.template_matcher = TemplateMatcher(template_dir, sign_templates, template_scales, template_threshold)

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.voice_pub = rospy.Publisher(self.voice_topic, String, queue_size=20)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2 ** 24)
        if self.use_laser:
            rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)

        self.say("team_info", self.team_name)
        rospy.loginfo("city_roads_controller started")

    def say(self, key, payload=""):
        cache_key = "%s::%s" % (key, payload)
        current = now_sec()
        if current - self.last_voice.get(cache_key, 0.0) < self.voice_cooldown_seconds:
            return
        self.last_voice[cache_key] = current
        msg = String()
        msg.data = cache_key
        self.voice_pub.publish(msg)

    def set_mode(self, name, seconds, linear_speed, angular_speed):
        self.active_mode = name
        self.mode_until = now_sec() + seconds
        self.mode_linear = linear_speed
        self.mode_angular = angular_speed

    def clear_mode(self):
        self.active_mode = None
        self.mode_until = 0.0
        self.mode_linear = 0.0
        self.mode_angular = 0.0

    def scan_callback(self, msg):
        front_samples = []
        angle_limit = math.radians(self.laser_front_angle_deg)
        angle = msg.angle_min
        for distance in msg.ranges:
            if abs(angle) <= angle_limit and msg.range_min < distance < msg.range_max:
                front_samples.append(distance)
            angle += msg.angle_increment
        self.front_min_range = min(front_samples) if front_samples else float("inf")

    def image_callback(self, msg):
        self.frame_index += 1
        if self.frame_skip > 1 and (self.frame_index % self.frame_skip) != 0:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cmd, debug_frame = self.process_frame(frame)
        self.cmd_pub.publish(cmd)
        if self.publish_debug_image and debug_frame is not None:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8"))

    def process_frame(self, frame):
        debug = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        line_error, line_found = self.detect_lane(frame, hsv, debug)
        light_state = self.detect_traffic_light(hsv, debug)
        crosswalk_detected = self.detect_crosswalk(frame, debug)
        cone_direction = self.detect_cone(hsv, debug)
        finish_detected = self.detect_finish_line(frame, debug)

        if self.template_matcher.templates and (self.frame_index % self.template_detect_every_n_frames == 0):
            sign_label, score, box = self.template_matcher.detect(frame, self.template_roi)
            if box is not None:
                x, y, width, height = box
                cv2.rectangle(debug, (x, y), (x + width, y + height), (255, 0, 0), 2)
                cv2.putText(debug, "%s %.2f" % (sign_label, score), (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            if sign_label:
                self.handle_sign(sign_label)

        if self.finish_done:
            return Twist(), debug

        current = now_sec()
        if self.active_mode and current >= self.mode_until:
            self.clear_mode()

        if light_state == "red":
            self.pending_stop_reason = "red_light"
            self.say("red_light")
        elif light_state == "green" and self.pending_stop_reason == "red_light":
            self.pending_stop_reason = None
            self.say("green_light")

        if self.use_laser and self.front_min_range <= self.laser_front_stop_distance:
            self.pending_stop_reason = "pedestrian"
            self.say("pedestrian")
            cv2.putText(debug, "WAIT PEDESTRIAN", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return Twist(), debug

        if self.pending_stop_reason == "red_light":
            cv2.putText(debug, "RED LIGHT STOP", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return Twist(), debug

        if crosswalk_detected and current - self.last_crosswalk_time > self.crosswalk_cooldown_seconds:
            self.last_crosswalk_time = current
            self.set_mode("crosswalk_stop", self.crosswalk_stop_seconds, 0.0, 0.0)
            self.say("crosswalk")

        if cone_direction and current - self.last_cone_time > self.cone_cooldown_seconds:
            self.last_cone_time = current
            turn = -self.cone_avoid_turn if cone_direction == "left" else self.cone_avoid_turn
            self.set_mode("cone_avoid", self.cone_avoid_seconds, self.cone_avoid_speed, turn)
            self.say("cone")

        if self.danger_seen_time is not None and finish_detected:
            self.say("finish")
            self.finish_done = True
            self.set_mode("finish_roll", self.finish_roll_seconds, self.finish_speed, 0.0)

        if self.active_mode:
            cmd = Twist()
            cmd.linear.x = self.mode_linear
            cmd.angular.z = self.mode_angular
            cv2.putText(debug, self.active_mode, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            return cmd, debug

        speed = self.limited_speed if self.speed_mode == "limited" else self.cruise_speed
        cmd = Twist()
        if not line_found:
            cmd.linear.x = speed * 0.5
            cmd.angular.z = self.line_search_turn_speed
            cv2.putText(debug, "SEARCH LINE", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return cmd, debug

        d_error = line_error - self.last_line_error
        angular = -(self.line_kp * line_error + self.line_kd * d_error)
        angular = clamp(angular, -self.max_angular_speed, self.max_angular_speed)
        self.last_line_error = line_error

        cmd.linear.x = speed
        cmd.angular.z = angular
        cv2.putText(debug, "MODE %s" % self.speed_mode.upper(), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug, "ERR %.1f" % line_error, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return cmd, debug

    def detect_lane(self, frame, hsv, debug):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = ratio_to_rect(width, height, self.line_roi)
        roi_hsv = hsv[y1:y2, x1:x2]
        mask = hsv_mask(roi_hsv, self.line_hsv_lower, self.line_hsv_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours = find_contours(mask)

        best = None
        best_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.line_search_min_area:
                continue
            if area > best_area:
                best = contour
                best_area = area

        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 1)
        if best is None:
            return self.last_line_error, False

        moments = cv2.moments(best)
        if moments["m00"] == 0:
            return self.last_line_error, False

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        cx_global = x1 + cx
        cy_global = y1 + cy
        error = (cx_global - (width // 2 + self.line_center_bias_px))
        cv2.circle(debug, (cx_global, cy_global), 8, (255, 0, 255), -1)
        cv2.line(debug, (width // 2 + self.line_center_bias_px, y1), (width // 2 + self.line_center_bias_px, y2), (255, 255, 0), 2)
        return error, True

    def detect_crosswalk(self, frame, debug):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = ratio_to_rect(width, height, self.crosswalk_roi)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.crosswalk_white_threshold, 255, cv2.THRESH_BINARY)
        row_ratio = np.mean(binary > 0, axis=1)
        groups = 0
        in_group = False
        for ratio in row_ratio:
            if ratio > self.crosswalk_min_row_ratio:
                if not in_group:
                    groups += 1
                    in_group = True
            else:
                in_group = False
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 1)
        if groups >= self.crosswalk_min_groups:
            cv2.putText(debug, "CROSSWALK", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return True
        return False

    def detect_traffic_light(self, hsv, debug):
        height, width = hsv.shape[:2]
        x1, y1, x2, y2 = ratio_to_rect(width, height, self.traffic_light_roi)
        roi = hsv[y1:y2, x1:x2]
        red_mask = cv2.bitwise_or(hsv_mask(roi, self.tl_red1_lower, self.tl_red1_upper),
                                  hsv_mask(roi, self.tl_red2_lower, self.tl_red2_upper))
        green_mask = hsv_mask(roi, self.tl_green_lower, self.tl_green_upper)
        red_area = int(np.sum(red_mask > 0))
        green_area = int(np.sum(green_mask > 0))
        state = "none"
        color = (200, 200, 200)
        if red_area > self.traffic_light_red_area and red_area > green_area:
            state = "red"
            color = (0, 0, 255)
        elif green_area > self.traffic_light_green_area and green_area > red_area:
            state = "green"
            color = (0, 255, 0)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(debug, "LIGHT %s" % state.upper(), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return state

    def detect_cone(self, hsv, debug):
        height, width = hsv.shape[:2]
        x1, y1, x2, y2 = ratio_to_rect(width, height, self.cone_roi)
        roi = hsv[y1:y2, x1:x2]
        mask = cv2.bitwise_or(hsv_mask(roi, self.cone_red1_lower, self.cone_red1_upper),
                              hsv_mask(roi, self.cone_red2_lower, self.cone_red2_upper))
        contours = find_contours(mask)
        if not contours:
            return None
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < self.cone_min_area:
            return None
        x, y, width_box, height_box = cv2.boundingRect(best)
        cx = x + width_box / 2.0
        direction = "left" if cx < (roi.shape[1] / 2.0) else "right"
        cv2.rectangle(debug, (x1 + x, y1 + y), (x1 + x + width_box, y1 + y + height_box), (0, 0, 255), 2)
        cv2.putText(debug, "CONE %s" % direction.upper(), (x1 + x, max(15, y1 + y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return direction

    def detect_finish_line(self, frame, debug):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = ratio_to_rect(width, height, self.finish_line_roi)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.finish_line_white_threshold, 255, cv2.THRESH_BINARY)
        ratio = float(np.mean(binary > 0))
        cv2.rectangle(debug, (x1, y1), (x2, y2), (180, 180, 255), 1)
        return ratio > self.finish_line_min_ratio

    def handle_sign(self, label):
        if label == "speed_limit":
            self.speed_mode = "limited"
            self.say("speed_limit")
        elif label == "speed_unlimit":
            self.speed_mode = "cruise"
            self.say("speed_unlimit")
        elif label == "turn_left":
            self.set_mode("turn_left", self.turn_seconds, self.turn_speed, self.turn_left_angular)
            self.say("turn_left")
        elif label == "turn_right":
            self.set_mode("turn_right", self.turn_seconds, self.turn_speed, self.turn_right_angular)
            self.say("turn_right")
        elif label == "straight":
            self.set_mode("straight", self.straight_seconds, self.turn_speed, 0.0)
            self.say("straight")
        elif label == "lane_change_left":
            self.set_mode("lane_change_left", self.lane_change_seconds, self.lane_change_speed, self.lane_change_angular)
            self.say("lane_change_left")
        elif label == "lane_change_right":
            self.set_mode("lane_change_right", self.lane_change_seconds, self.lane_change_speed, -self.lane_change_angular)
            self.say("lane_change_right")
        elif label == "lane_change":
            if self.lane_change_default_direction == "right":
                self.set_mode("lane_change_right", self.lane_change_seconds, self.lane_change_speed, -self.lane_change_angular)
                self.say("lane_change_right")
            else:
                self.set_mode("lane_change_left", self.lane_change_seconds, self.lane_change_speed, self.lane_change_angular)
                self.say("lane_change_left")
        elif label == "danger":
            self.danger_seen_time = now_sec()
            self.say("danger")


def main():
    rospy.init_node("city_roads_controller")
    CityRoadsController()
    rospy.spin()


if __name__ == "__main__":
    main()
