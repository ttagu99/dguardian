#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "caffeclassifier.h"
#include <QtWidgets>
#include <QSound>


#define CASE_PLAYONCE int(0)
#define CASE_PLAYFREQ int(1)

class QMovie;

using namespace std;
using namespace cv;
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    int m_outerCamTimerID;
    int m_innerCamTimerID;
    double m_scaleFactor;
    bool m_findLargestObject;
    bool m_filterRects;
    QMovie *pMovie;

    void PlayAnimation(const QString &fileName);
    void PlayAnimation(string &fileName);
    void PlayAnimation(string &fileName, string &soundFile);
    void PlayAnimation(string &fileName, string &soundFile, int nCase);
    void PlayAnimation(const QString &fileName, int nCase);
    map<int,int> m_mapTimer;
    Ptr<cuda::CascadeClassifier> cascade_frontface_default;
    Ptr<cuda::CascadeClassifier> cascade_sideface_default;
    Ptr<cuda::CascadeClassifier> cascade_half_body_default;
    CascadeClassifier cascade_hand_default;

    VideoCapture m_outerCap;
    VideoCapture m_innerCap;
    QImage putImage(const Mat& mat);

    void dispLT(string strMsg, Rect& rect, Mat& mat);
    void dispRT(string strMsg, Rect& rect, Mat& mat);
    void dispLT(string strMsg, vector<Rect>& rects, Mat& mat);
    void dispLT(Mat& mat);
    void dispRT(Mat& mat);
    Rect getLargestRect(vector<Rect> rects);
    vector<Rect> extractFace(Mat image);
    CaffeClassifier m_face_classifier;
    CaffeClassifier m_hand_classifier;

    int m_nVerificate;
    string m_strPreWho;
    string m_strPreCommand;
    string m_postManInfoFile;
    void OuterFunc();
    void InnerFunc();
    Mat m_meanOuter;
    Mat m_meanInner;

    string m_strOpenDoorPlay;
    string m_strOpenDoorSound;
    string m_strInnerOpenSound;
    string m_strInnerOpenPlay;
    string m_strInnerCloseSound;
    string m_strInnerClosePlay;

    string m_strCallPoliPlay;
    string m_strCallPoliSound;
    string m_strGetBack;
    string m_strGetBackSound;
    string m_strPostManPlay;
    string m_strPostManSound;
    string m_strQuestionSound;
    Mat m_ClosedDoorImg;
    string strMasterName ;
    string strCriminalName ;
    string strPostmanName ;

    string strCallCammand ;
    string strOpenCommand ;

    QSound *m_qsound;
    bool m_nPlay;
    int m_nOutWhoCnt;
    vector<Rect> m_innerRects;

    void funcNothing();
    void sendSms();

    int m_nCurPlayCount;
    int m_nTotPlayCount;


private slots:
    void StopPlay(void);
protected:
    void timerEvent(QTimerEvent *event);


};

#endif // MAINWINDOW_H
