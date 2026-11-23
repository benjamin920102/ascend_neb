cmake_minimum_required(VERSION 3.18)
# Match the user's Ascend PRL project: its ASC language frontend is installed by CANN.
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Ascend C target architecture")
find_package(ASC REQUIRED)
project(ascend_neb LANGUAGES C CXX ASC)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NEB_ROOT "${CMAKE_SOURCE_DIR}/vendor/neb" CACHE PATH "Upstream SilasEigen/neb source")
set(NEB_REF "${NEB_ROOT}/pow/ref")
foreach(_f nebhash2.c shares.c blake3_min.c chacha8.c nebhash2.h shares.h blake3_min.h)
  if(NOT EXISTS "${NEB_REF}/${_f}")
    message(FATAL_ERROR "Missing ${NEB_REF}/${_f}; run scripts/setup_reference.sh first")
  endif()
endforeach()
find_package(OpenMP REQUIRED COMPONENTS C)

# CANN toolkit layouts vary: CANN 8.5 and 9.0 ship distinct layouts.
set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root")
if(NOT ASCEND_HOME)
  set(ASCEND_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
endif()
find_path(ASCEND_ACL_INCLUDE_DIR acl/acl.h
  HINTS "${ASCEND_HOME}/include" "${ASCEND_HOME}/aarch64-linux/include"
        /usr/local/Ascend/ascend-toolkit/latest/include
        /usr/local/Ascend/cann-9.0.0/aarch64-linux/include)
find_library(ASCENDCL_LIBRARY NAMES ascendcl acl
  HINTS "${ASCEND_HOME}/lib64" "${ASCEND_HOME}/aarch64-linux/lib64"
        /usr/local/Ascend/ascend-toolkit/latest/lib64
        /usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64)
if(NOT ASCEND_ACL_INCLUDE_DIR OR NOT ASCENDCL_LIBRARY)
  message(FATAL_ERROR "AscendCL headers/libs not found. Source CANN set_env.sh and set ASCEND_HOME_PATH.")
endif()

# Real native device kernel, built with the .asc backend. NOT torch/JIT.
add_library(neb_digest SHARED kernel/neb_digest.asc)
set_target_properties(neb_digest PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(neb_digest PRIVATE ${ASCEND_ACL_INCLUDE_DIR})
target_link_libraries(neb_digest PRIVATE tiling_api register platform m dl pthread ${ASCENDCL_LIBRARY})
target_compile_options(neb_digest PRIVATE
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)

add_executable(ascend_neb
  src/main.cpp src/ascend_backend.cpp src/data.cpp src/protocol.cpp native/bridge.c
  ${NEB_REF}/nebhash2.c ${NEB_REF}/shares.c ${NEB_REF}/blake3_min.c ${NEB_REF}/chacha8.c)
target_include_directories(ascend_neb PRIVATE src ${NEB_REF} ${ASCEND_ACL_INCLUDE_DIR})
target_link_libraries(ascend_neb PRIVATE neb_digest ${ASCENDCL_LIBRARY} OpenMP::OpenMP_C pthread dl m)
set_target_properties(ascend_neb PROPERTIES BUILD_RPATH "$ORIGIN")
