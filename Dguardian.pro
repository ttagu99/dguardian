#-------------------------------------------------
#
# Project created by QtCreator 2017-08-01T20:54:12
#
#-------------------------------------------------

QT       += core gui
QT += multimedia
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Dguardian
TEMPLATE = app

INCLUDEPATH += /usr/include \
           /home/nvidia/caffe/include \
           /home/nvidia/caffe/build/include \
           /usr/include/boost \
           /usr/local/cuda/include

LIBS += -L"/usr/lib" \
        -L"/home/nvidia/caffe/build/lib" \
        -L"/usr/lib/aarch-linux-gnu" \
        -lopencv_core -lopencv_imgcodecs -lopencv_highgui \
        -lopencv_videoio \
        -lopencv_imgproc -lopencv_cudaimgproc \
        -lopencv_cudaobjdetect \
        -lopencv_objdetect \
        -lcaffe \
        -lboost_system \
        -lglog

SOURCES += main.cpp\
        mainwindow.cpp \
    caffeclassifier.cpp

HEADERS  += mainwindow.h \
    onboardgrab.h \
    cvimagewidget.h \
    caffeclassifier.h


FORMS    += mainwindow.ui

DISTFILES += \
    haarcascade_frontalface_default.xml \
    train.binaryproto \
    deploy.prototxt \
    synset_words.txt \
    hand.xml
