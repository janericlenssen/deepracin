
TEMPLATE = lib
win32:CONFIG += staticlib

#On Windows put lib file directly into Debug/Release folders
CONFIG(debug, debug|release) {
win32:TARGET = ../../Debug/deepracin
}
else {
win32:TARGET = ../../Release/deepracin
}
#On Linux the files are put into project folder and the GStreamer is told to find the files there
unix:TARGET = ../deepracin

DEPENDPATH += .
INCLUDEPATH += .

#QT must not link itself into this lib, otherwise the GStreamer could not use it:
QT -= core gui
CONFIG -= qt
#CONFIG += console

CONFIG(debug, debug|release) {
    DEFINES+=_DEBUG
    DEFINES+=_CRT_SECURE_NO_WARNINGS
}
else {
    DEFINES+=_RELEASE
}


win32 {
    INCLUDEPATH +=  $$PWD/../../Libs/GStreamer/include/glib-2.0 \
                    $$PWD/../../Libs/GStreamer/include/gstreamer-0.10/gst/base \
                    $$PWD/../../Libs/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc

    LIBS +=         -L$$PWD/../../Libs/GStreamer/lib \
                    -L$$PWD/../../Libs/GStreamer/bin \
                    -L$$PWD/../../Libs/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib \
                    -L$$PWD/../../Libs/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib/Win32

    CONFIG(debug, debug|release) {
        LIBS += -lglib-2.0 \
                -lOpenCL

    } else {
        LIBS +=  -lglib-2.0 \
                -lOpenCL
    }

    #Copy all OpenCL kernels to plugin dir and delete existing ptx code
    #QMAKE_POST_LINK += del /S/Q ..\\..\\Libs\\GStreamerMinimal\\plugins\\Kernels\\*.h &
    #QMAKE_POST_LINK += del /S/Q ..\\..\\Libs\\GStreamerMinimal\\plugins\\Kernels\\*.cl &
    #QMAKE_POST_LINK += del /S/Q ..\\TestingVirusDetection\\Kernels\\*.h &
    #QMAKE_POST_LINK += del /S/Q ..\\TestingVirusDetection\\Kernels\\*.cl &
    #QMAKE_POST_LINK += copy /Y Kernels\\*.h ..\\..\\Libs\\GStreamerMinimal\\plugins\\Kernels\\ &
    #QMAKE_POST_LINK += copy /Y Kernels\\*.cl ..\\..\\Libs\\GStreamerMinimal\\plugins\\Kernels\\ &
    #QMAKE_POST_LINK += copy /Y Kernels\\*.h ..\\TestingVirusDetection\\Kernels\\ &
    #QMAKE_POST_LINK += copy /Y Kernels\\*.cl ..\\TestingVirusDetection\\Kernels\\ &
    #QMAKE_POST_LINK += del /S/Q ..\\*.ptx &
    #QMAKE_POST_LINK += del /S/Q ..\\..\\Libs\\GStreamerMinimal\\plugins\\*.ptx &
    #QMAKE_POST_LINK += del /S/Q ..\\TestingVirusDetection\\*.ptx
}

unix {
    INCLUDEPATH +=  /usr/lib/glib-2.0/include \
                    /usr/lib/x86_64-linux-gnu/glib-2.0/include \
                    /usr/include/glib-2.0/ \
                    include/ \
                    /usr/local/cuda/include


    CONFIG(debug, debug|release) {
        LIBS += -lglib-2.0 \
                -lOpenCL
    } else {
        LIBS += -lglib-2.0 \
                -lOpenCL
    }

    #eval(USER = odroid)
    #{
    #    #For ARM board with Mali GPU:
    #    message("Adding the dependencies for ARM board with Mali GPU")
    #   INCLUDEPATH += /usr/lib/arm-linux-gnueabihf/glib-2.0/include/
    #  LIBS +=  -L/usr/lib/arm-linux-gnueabihf/mali-egl #\
    #             -lmali
    #}

    #Copy all OpenCL kernels to plugin dir and delete existing ptx code
    #QMAKE_POST_LINK += rm -f ../../Libs/GStreamerMinimal/plugins/Kernels/*.h &
    #QMAKE_POST_LINK += rm -f ../../Libs/GStreamerMinimal/plugins/Kernels/*.cl &
    #QMAKE_POST_LINK += rm -f ../TestingVirusDetection/Kernels/*.h &
    #QMAKE_POST_LINK += rm -f ../TestingVirusDetection/Kernels/*.cl &
    #QMAKE_POST_LINK += cp -f Kernels/*.h ../../Libs/GStreamerMinimal/plugins/Kernels/ &
    #QMAKE_POST_LINK += cp -f Kernels/*.cl ../../Libs/GStreamerMinimal/plugins/Kernels/ &
    #QMAKE_POST_LINK += cp -f Kernels/*.h ../TestingVirusDetection/Kernels/ &
    #QMAKE_POST_LINK += cp -f Kernels/*.cl ../TestingVirusDetection/Kernels/ &
    #QMAKE_POST_LINK += rm -f ../*.ptx &
    #QMAKE_POST_LINK += rm -f ../../Libs/GStreamerMinimal/plugins/*.ptx &
    #QMAKE_POST_LINK += rm -f ../TestingVirusDetection/*.ptx
}

#Input
HEADERS += \
    include/deepRACIN.h \
    include/dR_clwrap.h \
    include/dR_core.h \
    include/dR_parser.h \
    include/dR_types.h \
    include/dR_base.h \
    include/dR_nodes_misc.h \
    include/dR_nodes_fc.h \
    include/dR_nodes_conv2d.h \
    include/dR_nodes_pooling.h \
    include/dR_nodes_math.h \
    include/dR_nodes_transform.h \
    include/dR_nodes_filter.h \
    include/dR_nodes_bn.h

SOURCES += \
    src/deepRACIN.c \
    src/dR_clwrap.c \
    src/dR_core.c \
    src/dR_parser.c \
    src/dR_nodes_conv2d.c \
    src/dR_nodes_pooling.c \
    src/dR_nodes_fc.c \
    src/dR_nodes_misc.c \
    src/dR_nodes_math.c \
    src/dR_nodes_transform.c \
    src/dR_nodes_filter.c \
    src/dR_nodes_bn.c

OTHER_FILES += \
    Kernels/dR_conv2d.cl \
    Kernels/dR_fcl.cl \
    Kernels/dR_helper.cl \
    Kernels/dR_lrn.cl \
    Kernels/dR_pooling.cl \
    Kernels/dR_math.cl \
    Kernels/dR_winograd2.cl \
    Kernels/dR_transform.cl \
    Kernels/dR_conv2d1x1.cl






