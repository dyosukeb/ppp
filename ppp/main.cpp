#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#endif

int main(int argc, char *argv[])
{
    int time;
    cv::Mat img , gray;
    // カメラからのビデオキャプチャを初期化する
    cv::VideoCapture cap(0);
    
    //キャプチャ画像をRGBで取得
    cap.set( CV_CAP_PROP_FRAME_HEIGHT, 480 );
    cap.set( CV_CAP_PROP_FRAME_WIDTH, 640 );
    
    //カメラがオープンできない場合終了
    if( !cap.isOpened() )return -1;
    
    // ウィンドウを作成する
    cv::namedWindow( "camera", CV_WINDOW_AUTOSIZE);
    
    // 分類器の読み込み(顔、目)
    std::string cascadeName = "/usr/local/Cellar/opencv/2.4.4a/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier cascade;
    if(!cascade.load(cascadeName))return -1;
    
    std::string nested_cascadeName = "/usr/local/Cellar/opencv/2.4.4a/share/OpenCV/haarcascades/haarcascade_eye.xml";
    //メガネ用
    //std::string nested_cascadeName = "/usr/local/Cellar/opencv/2.4.4a/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
    cv::CascadeClassifier nested_cascade;
    if(!nested_cascade.load(nested_cascadeName))return -1;
    
    //scaleの値を用いて元画像を縮小、符号なし8ビット整数型，1チャンネル(モノクロ)の画像を格納する配列を作成
    double scale = 4.0;
    
    // スペースキーが押下されるまで、ループをくり返す
    while( cvWaitKey( 1 ) != 32 )
    {
        cap >> img;
        time = 0;
        // グレースケール画像に変換
        cv::cvtColor(img, gray, CV_BGR2GRAY);
        cv::Mat smallImg(cv::saturate_cast<int>(img.rows/scale), cv::saturate_cast<int>(img.cols/scale), CV_8UC1);
        // 処理時間短縮のために画像を縮小
        cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
        cv::equalizeHist( smallImg, smallImg);
        
        std::vector<cv::Rect> faces;
        /// マルチスケール（顔）探索xo
        // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
        cascade.detectMultiScale(smallImg, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        
        // 結果の描画
        std::vector<cv::Rect>::const_iterator r = faces.begin();
        for(; r != faces.end(); ++r)
        {
            cv::Point faceCenter;
            int radius;
            double distance;
            faceCenter.x = cv::saturate_cast<int>((r->x + r->width*0.5)*scale);
            faceCenter.y = cv::saturate_cast<int>((r->y + r->height*0.5)*scale);
            radius = cv::saturate_cast<int>((r->width + r->height)*0.25*scale);
            cv::circle( img, faceCenter, radius, cv::Scalar(80,80,255), 3, 8, 0 );
            
            cv::Mat smallImgROI = smallImg(*r);
            std::vector<cv::Rect> nestedObjects;
            
            //マルチスケール（目）検索
            //画像、出力矩形、縮小スケール、最低矩形数、（フラグ）、最小矩形
            nested_cascade.detectMultiScale(smallImgROI, nestedObjects,1.1,3,CV_HAAR_SCALE_IMAGE,cv::Size(10,10));
            
            //検出結果（目）の描画
            std::vector<cv::Rect>::const_iterator nr = nestedObjects.begin();
            for(; nr != nestedObjects.end(); ++nr){
                time++;
                cv::Point eyeCenter;
                int radius;
                eyeCenter.x = cv::saturate_cast<int>((r->x + nr ->x + nr -> width*0.5)*scale);
                eyeCenter.y = cv::saturate_cast<int>((r->y + nr ->y + nr -> height*0.5)*scale);
                radius = cv::saturate_cast<int>((nr -> width + nr -> height)*0.1*scale);
                cv::circle(img, eyeCenter, radius, cv::Scalar(80,255,80),3,8,0);
                
                //顔の中心と目の中心の距離を計算
                distance = sqrt((faceCenter.x - eyeCenter.x)*(faceCenter.x - eyeCenter.x) + (faceCenter.y - eyeCenter.y)*(faceCenter.y - eyeCenter.y));
                if(distance > 100) printf("lier\n%d",time);
                //距離を表示
                //std::cout << cv::format("distance = %d",(int)distance) << std::endl;
                
            }
            
        }
        
        cv::imshow( "camera", img );
        
    }
    
}

