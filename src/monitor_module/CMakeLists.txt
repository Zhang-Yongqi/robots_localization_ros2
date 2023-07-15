
AUX_SOURCE_DIRECTORY(. DIR_MONITOR)
add_library(monitor SHARED  ${DIR_MONITOR})
target_include_directories(monitor PUBLIC .)
target_link_libraries(monitor
    ${Boost_LIBRARIES}
    )
