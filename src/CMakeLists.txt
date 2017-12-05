#cmake_minimum_required(VERSION 2.8)
#project( surfgpu )
#find_package( OpenCV REQUIRED )
#add_executable( surfgpu surf_gpu.cpp )
#target_link_libraries( surfgpu ${OpenCV_LIBS} )



cmake_minimum_required(VERSION 2.8)
project( surfgpu )

FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(Boost COMPONENTS python3)

ENABLE_TESTING()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS}) # Deprecated but so convenient!


add_library( surfgpu SHARED surf_gpu.cpp )
# don't prepend wrapper library name with lib
# set_target_properties( surfgpu PROPERTIES PREFIX "" )
target_link_libraries( surfgpu ${OpenCV_LIBS} ${Boost_LIBRARIES} )
