#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
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
    int timerId;
    Ptr<cuda::CascadeClassifier> cascade_frontface_default;
    Ptr<cuda::CascadeClassifier> cascade_eye_glass_default;
    Ptr<cuda::CascadeClassifier> cascade_eye_default;
    Ptr<cuda::CascadeClassifier> cascade_frontface_tree;

    VideoCapture cap;
protected:
    void timerEvent(QTimerEvent *event);

};

#endif // MAINWINDOW_H
