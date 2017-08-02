#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
//#include "onboardgrab.h"

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    Mat img = imread("/home/nvidia/Pictures/test.png");
    QImage qimg(img.data,    img.cols, img.rows, QImage::Format_RGB888);

    ui->innerLabel->setPixmap(QPixmap::fromImage(qimg));
    //cv::imshow("dis",inputImg);
}

MainWindow::~MainWindow()
{
    delete ui;
}
