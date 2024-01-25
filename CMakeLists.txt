cmake_minimum_required(VERSION 3.10)

project(oneday C)


add_compile_options(-std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects -march=native -g  -gdwarf-4)
add_link_options(-std=gnu2x -fenable-matrix  -ffast-math -Weverything -fwrapv -fno-exceptions -fno-rtti -fpic -fPIC -fPIE -pie -fvisibility=hidden -static-pie -flto -fwhole-program -ffat-lto-objects -march=native -g  -gdwarf-4)

add_compile_options(
        -Wno-unsafe-buffer-usage
        $<$<CONFIG:DEBUG>:-Og>
        $<$<CONFIG:RELEASE>:-Ofast>
)

add_compile_definitions(
        $<$<CONFIG:DEBUG>:DEBUG>
        $<$<CONFIG:RELEASE>:NDEBUG>
)

include_directories(./)

#add_executable(mcmc src/mcmc.c src/mcmc.h)
#add_executable(sopt src/sopt.c src/sopt.h)
#add_executable(riscv src/riscv.c)
add_executable(tonygrad src/tonygrad.c src/tonygrad.h
)
