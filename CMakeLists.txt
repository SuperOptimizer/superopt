cmake_minimum_required(VERSION 3.10)

project(oneday C)



add_executable(oneday main.c common.h  day/4/mcmc.c day/4/mcmc.h
        )

target_compile_options(oneday PUBLIC -std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects -march=native -g3)
target_link_options(oneday PUBLIC -std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects -march=native -g3)

target_compile_options(oneday PUBLIC
        $<$<CONFIG:DEBUG>:-O0 -g3>
        $<$<CONFIG:RELEASE>:-Ofast -g3>
)

target_compile_definitions(oneday PUBLIC
        $<$<CONFIG:DEBUG>:DEBUG>
)

target_include_directories(oneday PUBLIC ./)