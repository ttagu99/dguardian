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

    string cascadeName = "/home/nvidia/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml";
    cascade_gpu = cuda::CascadeClassifier::create(cascadeName);
    timerId = startTimer(1000/3);
}

MainWindow::~MainWindow()
{
    killTimer(timerId);
    delete ui;
}

void MainWindow::timerEvent(QTimerEvent *event)
{
     CVImageWidget* imageWidget = new CVImageWidget();
     //ui->widget->set(imageWidget);
     this->setCentralWidget(imageWidget);
     // Load an image
     VideoCapture cap(1);
     //cap.set(CV_CAP_PROP_FPS, 25);

     cv::Mat image;
     //cascade.load(cascadeName);
     if (cuda::getCudaEnabledDeviceCount() == 0)
     {
         qDebug() << "No GPU found or the library is compiled without CUDA support";
     }
     cap >> image;
     cuda::GpuMat frame(image);
     cuda::GpuMat grayframe;

     cuda::cvtColor(frame, grayframe, CV_BGR2GRAY);
     cuda::equalizeHist(grayframe,grayframe);

     vector<cv::Rect> faces;

     double scaleFactor = 1.2;
     bool findLargestObject = true;
     bool filterRects = true;
     cuda::GpuMat  facesBuf_gpu;
     cascade_gpu->setFindLargestObject(findLargestObject);
     cascade_gpu->setScaleFactor(scaleFactor);
     cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
     cascade_gpu->detectMultiScale(grayframe, facesBuf_gpu);
     cascade_gpu->convert(facesBuf_gpu, faces);

     for(unsigned int facenum=0;facenum<faces.size();++facenum)
     {
        cv::rectangle(image,faces[facenum],Scalar(0,0,255));
     }

     imageWidget->showImage(image);
}
