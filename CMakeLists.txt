cmake_minimum_required(VERSION 3.10)
project(image_recognition)

# Encontre OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Encontre TensorFlow
find_package(TensorFlow REQUIRED)
include_directories(${TensorFlow_INCLUDE_DIRS})

# Defina o executável
add_executable(image_recognition main.cpp)

# Link das bibliotecas
target_link_libraries(image_recognition ${OpenCV_LIBS} ${TensorFlow_LIBRARIES})
