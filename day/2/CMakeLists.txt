cmake_minimum_required(VERSION 3.10)

project(oneday C)



add_executable(oneday oneday.c oneday.h)

target_compile_options(oneday PUBLIC -std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects)
target_link_options(oneday PUBLIC -std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects)

target_compile_options(oneday PUBLIC
        $<$<CONFIG:DEBUG>:-Og -g3>
        $<$<CONFIG:RELEASE>:-Ofast>
)

target_compile_definitions(oneday PUBLIC
        $<$<CONFIG:DEBUG>:DEBUG>
)
