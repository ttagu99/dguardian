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
#include <QScrollBar>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->centralWidget->setFixedSize(1340,1000);
    ui->textBrowser->setGeometry(680,520,640,480);
    ui->play_label->setGeometry(10,520,640,480);
    pMovie = new QMovie(this);
    //m_strOpenDoorSound = "../dguardian/a2002011001-e02.wav";
    m_postManInfoFile = "../dguardian/WhoAreYou.txt";
    m_nOutWhoCnt = 20;
    //m_qsound = new QSound(QString::fromStdString(m_strOpenDoorPlay));
    //m_qsound->play();
   // m_qsound->stop();
  //  QSound::play(QString::fromStdString(m_strOpenDoorSound));

    connect(pMovie,SIGNAL(frameChanged(int)),this,SLOT(StopPlay()));

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

    ui->textBrowser->append(QString::fromStdString(model_file));
    ui->textBrowser->append(QString::fromStdString(trained_file));
    ui->textBrowser->append(QString::fromStdString(mean_file));
    ui->textBrowser->append(QString::fromStdString(label_file));
    string model_file_hand = "../dguardian/deploy_hand.prototxt";
    string trained_file_hand= "../dguardian/caffenet_hand_iter_99000.caffemodel";
    string mean_file_hand= "../dguardian/train_hand.binaryproto";
    string label_file_hand= "../dguardian/synset_words_hand.txt";

    strMasterName = "Daewoo\r";
    strCriminalName = "samgi\r";
    strPostmanName = "junhyun\r";
    strCallCammand = "A\r";
    strOpenCommand = "C\r";

    ui->textBrowser->append(QString::fromStdString(model_file_hand));
    ui->textBrowser->append(QString::fromStdString(trained_file_hand));
    ui->textBrowser->append(QString::fromStdString(mean_file_hand));
    ui->textBrowser->append(QString::fromStdString(label_file_hand));


    m_strOpenDoorPlay= "../dguardian/giphy.gif";
    m_strOpenDoorSound= "../dguardian/a2002011001-e02.wav";
    m_strCallPoliPlay= "../dguardian/giphy.gif";
    m_strCallPoliSound = "../dguardian/a2002011001-e02.wav";
    m_strGetBack= "../dguardian/giphy.gif";
    m_strGetBackSound = "../dguardian/a2002011001-e02.wav";
    m_strPostManPlay = "../dguardian/giphy.gif";
    m_strPostManSound= "../dguardian/a2002011001-e02.wav";
    PlayAnimation(m_strOpenDoorPlay,m_strOpenDoorSound);


    ui->textBrowser->moveCursor(QTextCursor::End);


    const int nBatchSize = 1;
    int nGpuNum = 0;
    m_scaleFactor = 1.01;
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
    cascade_frontface_default->setMinObjectSize(Size(30,30));
    //cascade_frontface_default->setMaxNumObjects(2);
    cascade_frontface_default->setMaxObjectSize(Size(500,400));


    string handDefaultXml = "../dguardian/hand.xml";
    cascade_hand_default = CascadeClassifier(handDefaultXml);

    m_outerCamTimerID = startTimer(1000/10);
    m_innerCamTimerID = startTimer(1000/10);
    m_mapTimer[m_outerCamTimerID] = 0;
    m_mapTimer[m_innerCamTimerID] = 1;

    m_outerCap = VideoCapture(1);
    m_outerCap.set(CV_CAP_PROP_FPS, 25);
    m_outerCap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    m_outerCap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    m_innerCap = VideoCapture(2);
    m_innerCap.set(CV_CAP_PROP_FPS, 25);
    m_innerCap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    m_innerCap.set(CV_CAP_PROP_FRAME_HEIGHT,480);


    
    unsigned int meanCnt = 10;


    Mat outer;
    Mat inner;

    for(unsigned int i=0;i<3;i++)
    {
        m_outerCap >> outer;
        m_innerCap >> inner;
    }
    m_outerCap >> m_meanOuter;
    m_innerCap >> m_meanInner;

    m_meanOuter /= meanCnt;
    m_meanInner /= meanCnt;
    for(unsigned int i=1;i<meanCnt;i++)
    {
        m_outerCap >> outer;
        m_innerCap >> inner;
        m_meanOuter += outer / meanCnt;
        m_meanInner += inner / meanCnt;
    }


    blur(m_meanOuter,m_meanOuter,Size(5,5));
    blur(m_meanInner,m_meanInner,Size(5,5));
    flip(m_meanOuter,m_meanOuter,1);
    flip(m_meanInner,m_meanInner,1);

    //imshow("meanOuter",m_meanOuter);

}

void MainWindow::StopPlay()
{
    if(pMovie->currentFrameNumber()>=0 && pMovie->frameCount()-1 ==pMovie->currentFrameNumber())
    {
        pMovie->stop();
        m_nPlay = false;
    }

}

MainWindow::~MainWindow()
{
    delete m_qsound;
    killTimer(m_outerCamTimerID);
    delete ui;
}

void MainWindow::funcNothing()
{
    Mat inImg;
    Mat outImg;
    m_innerCap >> inImg;
    flip(inImg,inImg,1);
    m_outerCap >> outImg;
    flip(outImg,outImg,1);
    dispLT(outImg);
    dispRT(inImg);
}

void MainWindow::InnerFunc()
{
    cv::Mat image;
    m_innerCap >> image;
    flip(image,image,1);
    vector<Rect> faceRects = extractFace(image);
    Rect faceRect = getLargestRect(faceRects);

    if(faceRect.width<30)
    {
        dispRT("Closer your face",faceRect,image);
        return;
    }
    int nThreshOutCnt = 10;
    if(m_nOutWhoCnt < nThreshOutCnt)
    {
        dispRT("Open Door",faceRect,image);
        m_nOutWhoCnt = 20;
        m_innerRects.push_back(faceRect);
        return;
    }

    int nCumFrame =20;
    int minX=9990;
    int maxX=0;
    int minY=9990;
    int maxY=0;

    if(m_innerRects.size()<nCumFrame)
    {
        m_innerRects.push_back(faceRect);
        dispRT("Do You Want Open Door?",faceRect,image);
        return;
    }

    for(size_t i=5;i<m_innerRects.size();i++)
    {
        if(minX > m_innerRects[i].x)
            minX = m_innerRects[i].x;

        if(maxX < m_innerRects[i].x)
            maxX = m_innerRects[i].x;

        if(minY > m_innerRects[i].y)
            minY = m_innerRects[i].y;

        if(maxY < m_innerRects[i].y)
            maxY = m_innerRects[i].y;

    }
    m_innerRects.clear();
    if((maxX-minX)>(maxY-minY)*1.2)
    {
        //NO
        dispRT("Not Open Door",m_innerRects[m_innerRects.size()-1],image);

    }
    else
    {
        //Yes
        dispRT("Open Door",m_innerRects[m_innerRects.size()-1],image);
        PlayAnimation(m_strOpenDoorPlay,m_strOpenDoorSound);
    }

    m_nOutWhoCnt = 0;

}

void MainWindow::OuterFunc()
{
    Mat image;
    m_outerCap >> image;
    flip(image,image,1);
//    Mat imageP;
//    Mat imageM;
//    imageP = image - m_meanOuter;
//    imageM = m_meanOuter - image;

//    Mat imageMax;
//    imageMax = cv::max(imageP,imageM);
//    Mat imageGray;
//    cvtColor(imageMax,imageGray,COLOR_RGB2GRAY);
//    Mat imageBin;
//    double thresh = 30.0;
//    threshold(imageGray,imageBin,thresh,1,CV_8UC1);
//    vector<Mat> splited_frame;
//    vector<Mat> bin_frame;
//    split(image, splited_frame);
//    for (size_t i = 0; i < splited_frame.size(); i++)
//    {
//      Mat temp  = splited_frame[i].mul(imageBin);
//      bin_frame.push_back(temp);
//    }
//    cv::merge(bin_frame,image);
    
    int nMinWidth = 50;
    float fMinProb = 0.99;
    int nCountThr = 5;

    vector<Rect> faceRects = extractFace(image);
    if(faceRects.size()>0)
    {
        m_nOutWhoCnt++;
    }
    else
    {
        m_nOutWhoCnt--;
    }

    string strWho;
    if(faceRects.size()>=2)
    {
        strWho = "Other People Stand Back";
        m_nVerificate = 0;
        m_strPreWho = "Other";
        dispLT(strWho,faceRects,image);
        PlayAnimation(m_strGetBack,m_strGetBackSound);
        return;
    }

    Rect faceRect = getLargestRect(faceRects);
    if(faceRect.width< nMinWidth)
    {
        strWho = "Please Close your Face";
        m_nVerificate=0;
        return dispLT(strWho,faceRect,image);
    }

    Mat imgFace = image(faceRect);
    vector<Prediction> v_who = m_face_classifier.ClassifyOverSample(imgFace,1,1);
    strWho = v_who.front().first;
    if(strWho!=m_strPreWho)
    {
        m_nVerificate = 0;
        m_strPreWho = strWho;
    }

    float fWhoProb = v_who.front().second;
    if(fWhoProb < fMinProb)
    {
        strWho = "Other People";
        m_nVerificate=0;
        dispLT(strWho, faceRect, image);
        return;
    }


    if(strWho == strPostmanName)
    {
        strWho = strPostmanName;
        if(m_strPreWho == strWho)
        {
            m_nVerificate++;
            std::ostringstream s;
            s <<  m_nVerificate*(100/nCountThr);
            std::string per(s.str());
            strWho += " : Face Verification Wait " + per + "%";
        }
        else
        {
            m_nVerificate = 0;
            m_strPreWho = strWho;
        }

        if(m_nVerificate>=nCountThr)
        {
            strWho += ": Please put in the mail box";
            ofstream outFile(m_postManInfoFile.c_str());
            outFile.close();
            PlayAnimation(m_strPostManPlay,m_strPostManSound);
        }

        return dispLT(strWho, faceRect, image);
    }


    if(strWho == strCriminalName)
    {
        strWho = strCriminalName;
        if(m_strPreWho == strWho)
        {
            m_nVerificate++;
            std::ostringstream s;
            s <<  m_nVerificate*(100/nCountThr);
            std::string per(s.str());
            strWho += " : Face Verification Wait " + per + "%";
            dispLT(strWho, faceRect, image);
        }
        else
        {
            m_nVerificate = 0;
            m_strPreWho = strWho;
        }

        if(m_nVerificate>=nCountThr)
        {
            strWho += ": Call Master !!";
            m_nVerificate =0;
            dispLT(strWho, faceRect, image);
            PlayAnimation(m_strCallPoliPlay,m_strCallPoliSound);
        }
        return;
    }

    if(strWho == strMasterName)
    {
        strWho = strMasterName;

        Rect rectHand = faceRect;
        string strHand;
        rectHand.x += rectHand.width + 15;
        rectHand.width *= 1.3;
        rectHand.height *= 1.3;

        if(rectHand.x + rectHand.width > image.cols)
            rectHand.width = image.cols - rectHand.x;

        if(rectHand.y + rectHand.height > image.rows)
            rectHand.height = image.rows - rectHand.y;

        rectangle(image,rectHand,Scalar(255,0,0),1);

        Mat imgHand=image(rectHand);
        //imshow("Debug", imgHand);
        //waitKey(0);
        vector<Prediction> v_hand = m_hand_classifier.ClassifyOverSample(imgHand,1,1);
        strHand = v_hand.front().first;
        float fHandProb = v_who.front().second;
        if(m_strPreWho == strWho && m_strPreCommand == strHand && fHandProb > fMinProb)
        {
            m_nVerificate++;
            std::ostringstream s;
            s <<  m_nVerificate*(100/nCountThr);
            std::string per(s.str());
            strWho += " : Face and Mot Verfi Wait" + per + "%";
        }
        else
        {
            m_nVerificate = 0;
            m_strPreWho = strWho;
            m_strPreCommand = strHand;
        }

        if(m_nVerificate>=nCountThr)
        {
            if(strHand == strCallCammand)
            {
               strWho += ": Call Poli And Open!!";
               dispLT(strWho, faceRect, image);
               PlayAnimation(m_strCallPoliPlay,m_strCallPoliSound);
            }
            else if(strHand == strOpenCommand)
            {
                strWho += ": Open !!";
                dispLT(strWho, faceRect, image);
                PlayAnimation(m_strOpenDoorPlay,m_strOpenDoorSound);
                //PlayAnimation(m_strOpenDoorPlay, "../dguardian/Casio-VZ-10M-Hard-Organ-C2.wav");
            }
            else
            {
                strWho += ": PW Fail" + strHand;
            }

            m_nVerificate = 0;
        }

        return dispLT(strWho, faceRect, image);
    }

    return dispLT(strWho, faceRect, image);
}

void MainWindow::dispRT(string strMsg, Rect& rect, Mat& mat)
{
    string strOut = "inner : " + strMsg;
    ui->textBrowser->append(QString::fromStdString(strOut));
    putText(mat,strMsg,rect.tl(),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),1.0);
    rectangle(mat,rect,Scalar(0,0,255),1);
    return dispRT(mat);
}
void MainWindow::dispLT(string strMsg, Rect& rect, Mat& mat)
{
    string strOut = "Outer : " + strMsg;
    ui->textBrowser->append(QString::fromStdString(strOut));
    putText(mat,strMsg,rect.tl(),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),1.0);
    rectangle(mat,rect,Scalar(0,0,255),1);
    return dispLT(mat);
}
void MainWindow::dispLT(string strMsg, vector<Rect>& rects, Mat& mat)
{
    ui->textBrowser->append(QString::fromStdString(strMsg));
    for(unsigned int i=0;i<rects.size();i++)
    {
        putText(mat,strMsg,rects[i].tl(),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),1.0);
        rectangle(mat,rects[i],Scalar(0,0,255),1);
    }
    return dispLT(mat);
}
void MainWindow::dispLT(Mat& mat)
{
   ui->label_outer->setPixmap(QPixmap::fromImage(putImage(mat)));
}

void MainWindow::dispRT(Mat& mat)
{
   ui->label_inner->setPixmap(QPixmap::fromImage(putImage(mat)));
}

void MainWindow::timerEvent(QTimerEvent *event)
{
    if(m_nPlay)
    {
        funcNothing();
        return;
    }
    int cur = event->timerId();
    switch(m_mapTimer[cur])
    {
        case 0:
        {
            OuterFunc();
            break;
        }
        case 1:
        {
            InnerFunc();
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
 void MainWindow::PlayAnimation(string &fileName)
 {
     PlayAnimation(QString::fromStdString(fileName));
 }
 void MainWindow::PlayAnimation(string &fileName, string &soundFile)
 {
     PlayAnimation(QString::fromStdString(fileName));
     //m_qsound->setObjectName(QString::fromStdString(soundFile));
     //m_qsound->play();
     QSound::play(QString::fromStdString(soundFile));
 }

void MainWindow::PlayAnimation(const QString &fileName)
{
    m_nPlay = true;
    pMovie->stop();
    ui->play_label->setMovie(pMovie);
    ui->play_label->setScaledContents(true);
    pMovie->setFileName(fileName);
    pMovie->start();

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
