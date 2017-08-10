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
    timerId = startTimer(1000/24);
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

     cv::cuda::CascadeClassifier cascade;
     //cascade.setMaxObjectSize(Size(300,300));
     string cascadeName = "haarcascade_frontalface_default.xml";
     cascade.load(cascadeName);


     cap >> image;

     cuda::GpuMat frame(image);
     cuda::GpuMat grayframe;

     cuda::cvtColor(frame, grayframe, CV_BGR2GRAY);
     cuda::equalizeHist(grayframe,grayframe);

     vector<cv::Rect> faces;
     cascade.detectMultiScale(grayframe, faces);

     for(int i=0;i<faces.size();++i)
     {
        cv::rectangle(image,faces[i],Scalar(255,0,0));
     }  // retrieve all detected faces and draw rectangles for visualization

     imageWidget->showImage(image);
}
