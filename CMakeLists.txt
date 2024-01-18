cmake_minimum_required(VERSION 3.10)

project(oneday C)


add_compile_options(-std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects -march=native -g3)
add_link_options(-std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects -march=native -g3)

add_compile_options(
        $<$<CONFIG:DEBUG>:-O0>
        $<$<CONFIG:RELEASE>:-Ofast>
)

add_compile_definitions(
        $<$<CONFIG:DEBUG>:DEBUG>
        $<$<CONFIG:RELEASE>:NDEBUG>
)

include_directories(./)

add_executable(oneday main.c common.h  day/4/mcmc.c day/4/mcmc.h)
add_executable(sopt day/5/sopt.c day/5/sopt.h)
