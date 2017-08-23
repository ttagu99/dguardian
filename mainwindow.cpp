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
    string trained_file= "../dguardian/caffenet_face_iter_145000.caffemodel";
    string mean_file= "../dguardian/train.binaryproto";
    string label_file= "../dguardian/synset_words.txt";


    string model_file_hand = "../dguardian/deploy_hand.prototxt";
    string trained_file_hand= "../dguardian/caffenet_hand_iter_99000.caffemodel";
    string mean_file_hand= "../dguardian/train_hand.binaryproto";
    string label_file_hand= "../dguardian/synset_words_hand.txt";

    const int nBatchSize = 1;
    int nGpuNum = 0;
    m_scaleFactor = 1.02;
    m_findLargestObject = false;
    m_filterRects = true;


    m_face_classifier.loadModel(model_file, trained_file, mean_file, label_file, true, nBatchSize, nGpuNum);
    m_face_classifier.setFcn(false);

    m_hand_classifier.loadModel(model_file_hand, trained_file_hand, mean_file_hand, label_file_hand, true, nBatchSize, nGpuNum);
    m_hand_classifier.setFcn(false);

    string frontfaceDefaultXml = "../dguardian/haarcascade_frontalface_default.xml";
    cascade_frontface_default = cuda::CascadeClassifier::create(frontfaceDefaultXml);
    cascade_frontface_default->setFindLargestObject(m_findLargestObject);
    cascade_frontface_default->setScaleFactor(m_scaleFactor);
    cascade_frontface_default->setMinNeighbors(6);
    cascade_frontface_default->setMinObjectSize(Size(50,50));
    cascade_frontface_default->setMaxNumObjects(2);
    cascade_frontface_default->setMaxObjectSize(Size(500,400));


    string handDefaultXml = "../dguardian/hand.xml";
    cascade_hand_default = CascadeClassifier(handDefaultXml);

    m_outerCamTimerID = startTimer(1000/10);
    m_innerCamTimerID = startTimer(1000/10);
    m_mapTimer[m_outerCamTimerID] = 0;
    m_mapTimer[m_innerCamTimerID] = 1;

    m_outerCap = VideoCapture(0);
    m_outerCap.set(CV_CAP_PROP_FPS, 30);
    m_innerCap = VideoCapture(1);
    m_innerCap.set(CV_CAP_PROP_FPS, 30);
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
             flip(image,image,1);
             int nMinWidth = 50;
             float fMinProb = 0.99;
             int nCountThr = 2;

             vector<Rect> hands;
             cascade_hand_default.detectMultiScale(image,hands,1.01,30,0,Size(50,50),Size(300,300));
             for(unsigned int fi=0;fi<hands.size();fi++)
             {
                 //putText(image,strWho,hands[fi].tl(),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
                 rectangle(image,hands[fi],Scalar(255,0,0),1);
             }


             vector<Rect> faceRects = extractFace(image);
             string strWho;
             if(faceRects.size()>=2)
             {
                 strWho = "Other People Stand Back";
                 for(unsigned int fi=0;fi<faceRects.size();fi++)
                 {
                     putText(image,strWho,faceRects[fi].tl(),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),1.0);
                     rectangle(image,faceRects[fi],Scalar(0,0,255),1);
                     m_nVerificate = 0;
                     m_strPreWho = "Other";
                 }
                 break;
             }
             
             Rect faceRect = getLargestRect(faceRects);
             
             if(faceRect.width > nMinWidth)
             {
                 Mat imgFace = image(faceRect);
                 vector<Prediction> v_who = m_face_classifier.ClassifyOverSample(imgFace,1,1);
                 strWho = v_who.front().first;
                 if(strWho!=m_strPreWho)
                 {
                     m_nVerificate = 0;
                     m_strPreWho = strWho;
                 }
                 
                 float fWhoProb = v_who.front().second;

                 size_t nDaewoo = strWho.find("Daewoo");
                 size_t nSamgi = strWho.find("samgi");
                 size_t nJunhyun = strWho.find("junhyun");
                 if(nDaewoo==string::npos && nSamgi == string::npos
                         && nJunhyun == string::npos && fWhoProb < fMinProb)
                 {
                     strWho = "Others";
                     m_nVerificate=0;
                 }
                 else
                 {
                     m_nVerificate++;
                     if(m_nVerificate<nCountThr)
                     {
                        std::ostringstream s;
                        s <<  m_nVerificate*(100/nCountThr);
                        std::string per(s.str());
                        strWho += " : Face Verification Wait " + per + "%";                        
                     }
                     else
                     {
                        strWho += " : Face Verificate Complete ";

                        Rect rectHand = faceRect;
                        string strHand;
                        rectHand.x += rectHand.width;
                        if(rectHand.x+rectHand.width>image.cols)
                        {
                            strHand = " : Close Hand";
                        }
                        else
                        {
                            Mat imgHand=image(rectHand);

                            vector<Prediction> v_hand = m_hand_classifier.ClassifyOverSample(imgHand,1,1);
                            strHand = v_hand.front().first;
                            if(strHand == "C\r")
                            {
                               strWho += " : PassWord Complete";
                            }
                            else if(strHand == "A\r")
                            {
                                strWho += " : Call Police";
                            }
                            else
                            {
                                strWho += " : " + strHand;
                            }
                        }
                     }
                 }
                 putText(image,strWho,faceRect.tl(),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),1.0);
                 rectangle(image,faceRect,Scalar(0,0,255),1);
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
            flip(image,image,1);
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

    cuda::GpuMat  Buf_gpu;

    cascade_frontface_default->detectMultiScale(grayframe, Buf_gpu);
    cascade_frontface_default->convert(Buf_gpu, faces);

    return faces;
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
