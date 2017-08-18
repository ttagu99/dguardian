#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include "cvimagewidget.h"
#include <caffe/caffe.hpp>

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    if (cuda::getCudaEnabledDeviceCount() == 0)
    {
        qDebug() << "No GPU found or the library is compiled without CUDA support";
    }

    string frontfaceDefaultXml = "../dguardian/haarcascade_frontalface_default.xml";
    cascade_frontface_default = cuda::CascadeClassifier::create(frontfaceDefaultXml);

    m_outerCamTimerID = startTimer(1000/25);
    m_innerCamTimerID = startTimer(1000/25);
    m_mapTimer[m_outerCamTimerID] = 0;
    m_mapTimer[m_innerCamTimerID] = 1;

    m_outerCap = VideoCapture(0);
    m_outerCap.set(CV_CAP_PROP_FPS, 25);

}

MainWindow::~MainWindow()
{
    killTimer(m_outerCamTimerID);
    delete ui;
}

void MainWindow::timerEvent(QTimerEvent *event)
{
    int cur = event->timerId();
    switch(m_mapTimer[cur])
    {
        case 0:
        {
             cv::Mat image;

             m_outerCap >> image;
             Rect faceRect = extractFace(image);
             if(faceRect.width > 100)
                rectangle(image,faceRect,Scalar(0,0,255));

             ui->label->setPixmap(QPixmap::fromImage(putImage(image)));
            break;
        }
        default :
        {
                break;
        }
    }
}

Rect MainWindow::getLargestRect(vector<Rect> rects)
{
    if(rects.size()==0)
    {
        Rect nullRect;
        nullRect.x = 0;
        nullRect.y = 0;
        nullRect.width = 0;
        nullRect.height = 0;
        return nullRect;
    }

    int maxwidth = 0;
    int retidx = 0;
    for(unsigned int idx = 0;idx<rects.size();idx++)
    {
        if(maxwidth < rects[idx].width )
        {
            maxwidth = rects[idx].width;
            retidx = idx;
        }
    }
    return rects[retidx];
}

Rect MainWindow::extractFace(Mat image)
{
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

    return getLargestRect(faces);
}

QImage MainWindow::putImage(const Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS=1
    if(mat.type()==CV_8UC1)
    {
        // Set the coloouterCamTimerIDr table (used to translate colour indexes to qRgb values)
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
