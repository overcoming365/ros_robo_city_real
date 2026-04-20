#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import shlex
import subprocess
import time

import rospy
from std_msgs.msg import String


def parse_event(raw_text):
    parts = raw_text.split("::", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return raw_text, ""


class CityRoadsVoice(object):
    def __init__(self):
        self.backend = rospy.get_param("~voice_backend", "topic")
        self.voice_command = rospy.get_param("~voice_command", "espeak-ng -v zh")
        self.voice_cooldown = float(rospy.get_param("~voice_cooldown_seconds", 1.5))
        self.last_spoken = {}

        self.event_text = {
            "team_info": u"队伍信息 %s",
            "turn_left": u"识别到左转标志",
            "turn_right": u"识别到右转标志",
            "straight": u"识别到直行标志",
            "crosswalk": u"识别到人行横道，停车观察",
            "speed_limit": u"识别到限速标志，降低速度",
            "speed_unlimit": u"识别到解除限速标志，恢复速度",
            "red_light": u"识别到红灯，停车等待",
            "green_light": u"识别到绿灯，允许通行",
            "lane_change_left": u"识别到左变道标志，开始左变道",
            "lane_change_right": u"识别到右变道标志，开始右变道",
            "lane_change": u"识别到变道标志，开始变道",
            "pedestrian": u"前方有行人，请等待",
            "cone": u"识别到红色锥桶，开始避让",
            "danger": u"识别到危险标志，准备通过终点",
            "finish": u"比赛完成",
        }

        rospy.Subscriber("/city_roads/voice", String, self.voice_callback, queue_size=20)

    def event_to_text(self, key, payload):
        if key in self.event_text:
            template = self.event_text[key]
            if "%s" in template:
                return template % payload
            return template
        if payload:
            return payload
        return key

    def should_skip(self, key, payload):
        current = time.time()
        cache_key = "%s::%s" % (key, payload)
        last_time = self.last_spoken.get(cache_key, 0.0)
        if current - last_time < self.voice_cooldown:
            return True
        self.last_spoken[cache_key] = current
        return False

    def speak(self, text):
        if self.backend == "none":
            return
        if self.backend == "topic":
            rospy.loginfo("VOICE %s", text)
            return
        if self.backend == "espeak":
            cmd = shlex.split(self.voice_command)
            cmd.append(text.encode("utf-8") if not isinstance(text, str) else text)
            try:
                subprocess.call(cmd)
            except Exception as exc:
                rospy.logwarn("Voice backend failed: %s", exc)
            return
        rospy.logwarn("Unknown voice backend: %s", self.backend)

    def voice_callback(self, msg):
        key, payload = parse_event(msg.data)
        if self.should_skip(key, payload):
            return
        self.speak(self.event_to_text(key, payload))


def main():
    rospy.init_node("city_roads_voice")
    CityRoadsVoice()
    rospy.spin()


if __name__ == "__main__":
    main()
