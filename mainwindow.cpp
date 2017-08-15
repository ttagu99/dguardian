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
    string frontfaceDefaultXml = "../haarcascade_frontalface_default.xml";
    cascade_frontface_default = cuda::CascadeClassifier::create(frontfaceDefaultXml);

    timerId = startTimer(1000/25);

    cap = VideoCapture(1);
    cap.set(CV_CAP_PROP_FPS, 25);

}

MainWindow::~MainWindow()
{
    killTimer(timerId);
    delete ui;
}

void MainWindow::timerEvent(QTimerEvent *event)
{
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

     double scaleFactor = 1.1;
     bool findLargestObject = false;
     bool filterRects = true;
     cuda::GpuMat  Buf_gpu;
     cascade_frontface_default->setFindLargestObject(findLargestObject);
     cascade_frontface_default->setScaleFactor(scaleFactor);
     cascade_frontface_default->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
     cascade_frontface_default->detectMultiScale(grayframe, Buf_gpu);
     cascade_frontface_default->convert(Buf_gpu, faces);

     for(unsigned int idx=0;idx<faces.size();++idx)
     {
        cv::rectangle(image,faces[idx],Scalar(0,0,255));
     }

     ui->label->setPixmap(QPixmap::fromImage(putImage(image)));
}


QImage MainWindow::putImage(const Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS=1
    if(mat.type()==CV_8UC1)
    {
        // Set the color table (used to translate colour indexes to qRgb values)
        QVector<QRgb> colorTable;
        for (int i=0; i<256; i++)
            colorTable.push_back(qRgb(i,i,i));
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
        img.setColorTable(colorTable);
        return img;
    }
    // 8-bits unsigned, NO. OF CHANNELS=3
    if(mat.type()==CV_8UC3)
    {
        // Copy input Mat
        const uchar *qImageBuffer = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}
