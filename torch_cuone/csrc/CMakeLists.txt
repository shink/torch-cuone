# csrc
file(GLOB_RECURSE _PY_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
LIST(APPEND PY_SRC ${_PY_SRC})

# Pass to parent
set(PY_SRC ${PY_SRC} PARENT_SCOPE)
