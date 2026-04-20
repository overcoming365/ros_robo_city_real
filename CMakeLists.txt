cmake_minimum_required(VERSION 2.8.3)
project(city_roads)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/city_roads_controller.py
  scripts/city_roads_voice.py
  scripts/city_roads_hsv_tool.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config templates
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
