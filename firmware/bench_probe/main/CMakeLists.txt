include(${CMAKE_CURRENT_LIST_DIR}/models.cmake)
idf_component_register(SRCS "main.cc"
                       INCLUDE_DIRS "."
                       EMBED_FILES ${BENCH_MODELS})
