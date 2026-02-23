cmake_minimum_required(VERSION 3.10)
project(lerobot_dataset_v2_recorder_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROS 2 ament packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(vla_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_srvs REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# Apache Arrow + Parquet
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# nlohmann JSON
find_package(nlohmann_json REQUIRED)

add_library(recorder_node SHARED src/recorder_node.cpp)

ament_target_dependencies(recorder_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  image_transport
  vla_msgs
  ament_index_cpp
  std_srvs
)

target_link_libraries(recorder_node
  ${OpenCV_LIBS}
  Arrow::arrow_shared
  Parquet::parquet_shared
  nlohmann_json::nlohmann_json
)

rclcpp_components_register_node(recorder_node
  PLUGIN "LeRobotRecorderNode"
  EXECUTABLE recorder_node_exec
)

install(TARGETS recorder_node recorder_node_exec
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY resources/
  DESTINATION share/${PROJECT_NAME}/resources
)

ament_package()
