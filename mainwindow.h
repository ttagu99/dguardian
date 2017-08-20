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

    map<int,int> m_mapTimer;
    Ptr<cuda::CascadeClassifier> cascade_frontface_default;

    VideoCapture m_outerCap;

    QImage putImage(const Mat& mat);
    
    Rect getLargestRect(vector<Rect> rects);
    Rect extractFace(Mat image);
    CaffeClassifier face_classifier;

protected:
    void timerEvent(QTimerEvent *event);


};

#endif // MAINWINDOW_H
