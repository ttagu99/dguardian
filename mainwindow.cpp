#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include "cvimagewidget.h"
using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timerId = startTimer(1000);
}

MainWindow::~MainWindow()
{
    killTimer(timerId);
    delete ui;
}

void MainWindow::timerEvent(QTimerEvent *event)
{
     CVImageWidget* imageWidget = new CVImageWidget();
     this->setCentralWidget(imageWidget);
     // Load an image
     VideoCapture cap(1);
     cv::Mat image;

     cap >> image;
     imageWidget->showImage(image);


}
