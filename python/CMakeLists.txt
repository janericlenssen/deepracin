find_program(PYTHON "python")

if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    set(DEPS_IN     "${CMAKE_CURRENT_SOURCE_DIR}/deepracin/__init__.py.in")
    set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/deepracin/__init__.py")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})
    configure_file(${DEPS_IN} ${DEPS})

    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON} ${SETUP_PY} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})

    add_custom_target(target ALL DEPENDS ${OUTPUT})

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()

