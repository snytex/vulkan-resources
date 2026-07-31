cmake_minimum_required(VERSION 3.20)
project(vk_space LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/vendor/imgui)

add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/imgui_demo.cpp
  ${IMGUI_DIR}/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR})
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

add_executable(vk_space src/main.cpp)
target_include_directories(vk_space PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vk_space PRIVATE imgui Vulkan::Vulkan glfw glm::glm)
target_compile_options(vk_space PRIVATE -Wall -Wextra -Wno-missing-field-initializers)

if(NOT Vulkan_GLSLC_EXECUTABLE)
  find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc REQUIRED)
endif()

set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
  ${SHADER_SOURCE_DIR}/*.vert
  ${SHADER_SOURCE_DIR}/*.frag
  ${SHADER_SOURCE_DIR}/*.comp
  ${SHADER_SOURCE_DIR}/*.geom
)

foreach(shader IN LISTS SHADER_SOURCES)
  get_filename_component(shader_name ${shader} NAME)
  set(spv ${SHADER_BINARY_DIR}/${shader_name}.spv)
  add_custom_command(
    OUTPUT ${spv}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${shader} -o ${spv}
    DEPENDS ${shader}
    COMMENT "Compiling ${shader_name}"
    VERBATIM
  )
  list(APPEND SPV_FILES ${spv})
endforeach()

add_custom_target(shaders DEPENDS ${SPV_FILES})
add_dependencies(vk_space shaders)
