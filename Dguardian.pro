#-------------------------------------------------
#
# Project created by QtCreator 2017-08-01T20:54:12
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Dguardian
TEMPLATE = app

INCLUDEPATH += /usr/include/opencv
LIBS += -L /usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio
LIBS += -L /usr/lib -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudaobjdetect

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    onboardgrab.h \
    cvimagewidget.h

FORMS    += mainwindow.ui

DISTFILES += \
    haarcascade_frontalface_default.xml
