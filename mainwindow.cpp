#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include "cvimagewidget.h"
#include <caffe/caffe.hpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>


using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->label_outer_dscript->setText("Outer Camera");
    ui->label_inner_dscript->setText("Inner Camera");
    QFont font = ui->label_outer_dscript->font();
    font.setPointSize(19);
    font.setBold(true);
    ui->label_outer_dscript->setFont(font);
    ui->label_inner_dscript->setFont(font);

    m_nVerificate =0;

    if (cuda::getCudaEnabledDeviceCount() == 0)
    {
        qDebug() << "No GPU found or the library is compiled without CUDA support";
    }

    string model_file = "../dguardian/deploy.prototxt";
    string trained_file= "../dguardian/caffenet_face_iter_110000.caffemodel";
    string mean_file= "../dguardian/train.binaryproto";
    string label_file= "../dguardian/synset_words.txt";

    const int nBatchSize = 1;
    const int nOverSample = 1;
    int nGpuNum = 0;


    m_face_classifier.loadModel(model_file, trained_file, mean_file, label_file, true, nBatchSize, nGpuNum);
    m_face_classifier.setFcn(false);

    string frontfaceDefaultXml = "../dguardian/haarcascade_frontalface_default.xml";
    cascade_frontface_default = cuda::CascadeClassifier::create(frontfaceDefaultXml);

    m_outerCamTimerID = startTimer(1000/6);
    m_innerCamTimerID = startTimer(1000/6);
    m_mapTimer[m_outerCamTimerID] = 0;
    m_mapTimer[m_innerCamTimerID] = 1;

    m_outerCap = VideoCapture(0);
    m_outerCap.set(CV_CAP_PROP_FPS, 25);

    m_innerCap = VideoCapture(1);
    m_innerCap.set(CV_CAP_PROP_FPS, 25);
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
             vector<Rect> faceRects = extractFace(image);
             string strWho;
             if(faceRects.size()>=2)
             {
                 strWho = "Other People Stand Back";
                 for(int fi=0;fi<faceRects.size();fi++)
                 {
                     putText(image,strWho,faceRects[fi].tl(),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
                     rectangle(image,faceRects[fi],Scalar(0,0,255),3);
                 }
             }
             
             Rect faceRect = getLargestRect(faceRects);
             
             if(faceRect.width > 50)
             {
                 Mat imgFace = image(faceRect);
                 vector<Prediction> v_who = m_face_classifier.ClassifyOverSample(imgFace,1,1);
                 strWho = v_who.front().first;
                 
                 float fWhoProb = v_who.front().second;

                 size_t nDaewoo = strWho.find("Daewoo");
                 size_t nSamgi = strWho.find("samgi");
                 size_t nJunhyun = strWho.find("junhyun");
                 if(nDaewoo==string::npos && nSamgi == string::npos
                         && nJunhyun == string::npos && fWhoProb < 0.98)
                 {
                     strWho = "Others";
                     m_nVerificate=0;
                 }
                 else
                 {
                     m_nVerificate++;
                     if(m_nVerificate<5)
                     {
                        std::ostringstream s;
                        s <<  m_nVerificate*20;
                        std::string per(s.str());
                        strWho += " : Processing Wait " + per + "%";
                     }
                     else
                     {
                        strWho += " : Verificate Complete ";
                        //m_nVerificate =0;
                     }
                 }
                 putText(image,strWho,faceRect.tl(),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
                 rectangle(image,faceRect,Scalar(0,0,255),3);
             }
             else
             {
                 m_nVerificate=0;
             }
             ui->label_outer->setPixmap(QPixmap::fromImage(putImage(image)));
            break;
        }
        case 1:
        {
            cv::Mat image;
            m_innerCap >> image;
            vector<Rect> faceRects = extractFace(image);
            Rect faceRect = getLargestRect(faceRects);
            if(faceRect.width > 100)
            {
                rectangle(image,faceRect,Scalar(0,0,255),3);
            }
            ui->label_inner->setPixmap(QPixmap::fromImage(putImage(image)));
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

vector<Rect> MainWindow::extractFace(Mat image)
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

    return faces;
    //return getLargestRect(faces);
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
