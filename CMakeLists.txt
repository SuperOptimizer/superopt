cmake_minimum_required(VERSION 3.10)

project(oneday C)



add_executable(oneday day/3/riscv.c day/3/oneday.c day/3/oneday.h)

target_compile_options(oneday PUBLIC -std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects -march=native)
target_link_options(oneday PUBLIC -std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects -march=native)

target_compile_options(oneday PUBLIC
        $<$<CONFIG:DEBUG>:-Og -g3>
        $<$<CONFIG:RELEASE>:-Ofast>
)

target_compile_definitions(oneday PUBLIC
        $<$<CONFIG:DEBUG>:DEBUG>
)
