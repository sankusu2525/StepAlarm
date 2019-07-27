package com.sankusu2525.stepalarm;

import android.content.DialogInterface;
import android.media.AudioAttributes;
import android.media.AudioManager;
import android.media.SoundPool;
import android.os.Build;
import android.os.Vibrator;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

import static java.lang.Math.abs;
import static org.opencv.core.Core.calcCovarMatrix;
import static org.opencv.core.Core.fastAtan2;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.video.Video.OPTFLOW_FARNEBACK_GAUSSIAN;
import static org.opencv.video.Video.OPTFLOW_LK_GET_MIN_EIGENVALS;
import static org.opencv.video.Video.OPTFLOW_USE_INITIAL_FLOW;

import static java.lang.Math.sqrt;
import static java.lang.Math.pow;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";
    private final boolean LogOutput =false;

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean  mIsJavaCamera     = true;
    private Mat mGray_prev, mGray_next, mGray_Draw;
    MatOfPoint2f prev_pts;
    Point [] prevPoint
            ,AlertRect
            ,prevCorner;
    SettingIO SettingValue ;

    static {                                   // <-
        System.loadLibrary("opencv_java3");    // <- この3行を追加
    }                                          // <-
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");
                    mOpenCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
//    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
        if(!OpenCVLoader.initDebug()) Log.i("OpenCV", "Failed");
        else Log.i("OpenCV", "successfully built !");

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        SetSoundPool();
        SettingValue = new SettingIO(this,ArrayName.length);
        if(SettingValue.CreateCount ==0)
            mSetValue();
        else{
            for (int i = 0; i <ArrayName.length ; i++) {
                ArrayName[i].value = SettingValue.mArrayValue[i];
            }
            mSetValue();
        }
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
//            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        if(LogOutput)Log.d(TAG, "onCameraViewStarted");
        mGray_prev = new Mat();
        mGray_next = new Mat();
        mGray_Draw = new Mat();
    }

    public void onCameraViewStopped() {
        mGray_prev.release();
        mGray_next.release();
        mGray_Draw.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if(LogOutput)Log.d(TAG, "onCameraFrame");
        if(inputFrame ==null)return null;
        if(mGray_prev.empty()){
            mGray_prev = inputFrame.gray().clone();
            try {
                Thread.sleep((long)(1000 *WalkWait));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            return mGray_Draw;
        }
        mGray_next = inputFrame.gray().clone();
        if(mGray_next.empty()){
            return mGray_Draw;
        }

        Point center = new Point(mGray_prev.cols()/2, mGray_prev.rows()/2);
        MatOfPoint2f CameraCenter =new MatOfPoint2f();
        CameraCenter.fromArray(center);
        //Camera
        int mCameraAngle = CameraAngle ;//90.0f, CameraScale = 1.0f;

        //AlertLevel
        double mAlertLevel = AlertLevel
                , mRangeOverTime =RangeOverTime
                , mWalkLevel  = WalkLevel;
        int mAlertAreaCount =AlertAreaCount;
        int [] AlertArea =new int[3];

        //CheckPoint
        int mPtRow =25, mPtCol =25
                ,mPtCheck =PixelLevel ,mPtRowCheck ,mPtColCheck
                ,mPtAngle =PtAngle;
        Size mPt =new Size(mPtCol, mPtRow);
        Size mPtSize =new Size( PtSize_x, PtSize_y);
        Size mPtOffsetPoint   =new Size( center.x * PtOffsetPoint_x, center.y *PtOffsetPoint_y);
        Size mPtAn   =new Size( PtAn_x, PtAn_y);

        double mGaussianBlur =GaussianBlur;
        int imageErrer =-1
                ,mSafetyCount =(int)(mPtRow *mPtCol *SafetyLevel)
                ,mErrerLevel =(int)(mPtRow *mPtCol *ErrerLevel);

        MatOfPoint2f next_pts =new MatOfPoint2f();
        Video mVideo  =new Video();
        Size flowSize =new Size(mPtRow, mPtCol);

        if(prev_pts ==null || PtChange) {
            prev_pts   =new MatOfPoint2f();
            prevPoint =new Point[mPtRow * mPtCol];
            Size mPtAnPitch   =new Size((0.5f -mPtAn.width) *2 /mPt.width
                    , (0.5f -mPtAn.height) *2 /mPt.height);
            Size mPtAnReversePitch   =new Size(-((0.5f -mPtAn.width) *2 /mPt.width) /2
                    , -((0.5f -mPtAn.height) *2 /mPt.height) /2);
            Size mPtAnReverse   =new Size( (1.0f -mPtAn.width) /2, (1.0f -mPtAn.height) /2);

            for (int i = 0; i < flowSize.height; ++i) {
                for (int j = 0; j < flowSize.width; ++j) {
                    double cols = i * (float) mGray_prev.cols() / (flowSize.width - 1);
                    cols *=mPtAn.width +mPtAnPitch.width *j;
                    cols +=mGray_prev.cols() *(mPtAnReverse.width +mPtAnReversePitch.width *j);
                    cols = (cols - center.x) * mPtSize.width +mPtOffsetPoint.width;

                    double rows = j * (float) mGray_prev.rows() / (flowSize.height - 1);
                    rows *=mPtAn.height +mPtAnPitch.height *i;
                    rows +=mGray_prev.rows() *(mPtAnReverse.height +mPtAnReversePitch.height *i);
                    rows = (rows - center.y) *mPtSize.height +mPtOffsetPoint.height;
                    prevPoint[i *mPtCol +j] = new Point(cols, rows);
                }
            }
            prev_pts.fromArray(prevPoint);
            PtChange =false;
        }
        // Lucas-Kanadeメソッド＋画像ピラミッドに基づくオプティカルフロー
        // parameters=default
        Point [] nextPoint;
        MatOfByte Status =new MatOfByte();
        MatOfFloat Error =new MatOfFloat();
        Size WinSize =new Size(25,25);
        int maxLevel =3;
        int searchSize =30;
        TermCriteria criteria = new TermCriteria(TermCriteria.COUNT+TermCriteria.EPS, searchSize, 0.01f);
        int  	flags = OPTFLOW_LK_GET_MIN_EIGENVALS;
        double  minEigThreshold = 1e-4;

        mVideo.calcOpticalFlowPyrLK(mGray_prev, mGray_next, prev_pts, next_pts, Status, Error, WinSize, maxLevel, criteria, flags, minEigThreshold);
        nextPoint  =next_pts.toArray();

/*        if(LogOutput)
            Log.i(TAG, " calcOpticalFlowPyrLK: "
                    +" mGray_prev:" + mGray_prev.empty()
                    +" mGray_next:" + mGray_next.empty()
                    +" mVideo:" + mVideo.toString()
                    +" Status:" +Status.toString()
                    +" Error:" +Error.toString()
                    +" next_pts:" +next_pts.toString()
                    +" prevPoint:" +prevPoint[0].toString()
            );*/
        //createPrevCornerPt
        prevCorner =new Point[4];
        AlertRect =new Point[4];
        int Pt_i ;
        for (int i = 0; i <prevCorner.length ; i++) {
            switch (i){
                case 0:
                    Pt_i =0 *0                  ; break;
                case 1:
                    Pt_i =(mPtRow -1) *mPtCol   ; break;
                case 2:
                    Pt_i =mPtRow *mPtCol -1     ; break;
                case 3:
                    Pt_i =mPtCol -1             ; break;
                default:
                    Pt_i =0                     ; break;
            }
            prevCorner[i] =prevPoint[Pt_i].clone();
            AlertRect[i] =prevPoint[Pt_i].clone();
/*            if(LogOutput)
                Log.i(TAG, " prevCorner next xy: "
                        +" i:" +i
                        +" Pt_i:" +Pt_i
                        +" prevPoint:" +prevPoint[Pt_i]
                );*/
        }

/*        if(LogOutput)
            for (int i = 0; i <prevCorner.length ; i++) {
                Log.i(TAG, " prevCorner xy: "
                        +" i:" +i
                        +" prevCorner:" +prevCorner[i]
                );
            }*/

        //AlertRectCorner
        switch (mPtAngle){
            case 1:
                Pt_i =mPtCol /AlertArea.length ;
                AlertRect[3] =prevPoint[Pt_i ].clone();
                AlertRect[2] =prevPoint[(mPtRow -1) *mPtCol +Pt_i].clone();
                mPtRowCheck =(int)(prevCorner[2].x -prevCorner[3].x)/mPtCheck;
                mPtColCheck =(int)(prevCorner[3].y -prevCorner[0].y)/mPtCheck;
                break;
            case 2:
                Pt_i =mPtRow /AlertArea.length ;
                AlertRect[1] =prevPoint[Pt_i *mPtCol].clone();
                AlertRect[2] =prevPoint[Pt_i *mPtCol +mPtCol -1].clone();
                mPtColCheck =(int)(prevCorner[2].y -prevCorner[1].y)/mPtCheck;
                mPtRowCheck =(int)(prevCorner[1].x -prevCorner[0].x)/mPtCheck;
                break;
            case 3:
                Pt_i =mPtCol /AlertArea.length *(AlertArea.length -1);
                AlertRect[0] =prevPoint[Pt_i ].clone();
                AlertRect[1] =prevPoint[(mPtRow -1) *mPtCol +Pt_i].clone();
                mPtRowCheck =(int)(prevCorner[1].x -prevCorner[0].x)/mPtCheck;
                mPtColCheck =(int)(prevCorner[3].y -prevCorner[0].y)/mPtCheck;
                break;
            default:
                Pt_i =mPtRow /AlertArea.length *(AlertArea.length -1);
/*                if(LogOutput)
                    Log.i(TAG, " AlertRect Pt: "
                            +" mPtAngle:" +mPtAngle
                            +" Pt_i:" +Pt_i);*/
                AlertRect[0] =prevPoint[Pt_i *mPtCol].clone();
                AlertRect[3] =prevPoint[Pt_i *mPtCol +mPtCol -1].clone();
                mPtColCheck =(int)(prevCorner[3].y -prevCorner[0].y)/mPtCheck;
                mPtRowCheck =(int)(prevCorner[1].x -prevCorner[0].x)/mPtCheck;
                break;
        }


        //コーナー推測
//        Point [] nextCorner =RectCorner(nextPoint,mPtRow,mPtCol);
        Point [] nextCorner =RectCorner(nextPoint,Status,mPtRow,mPtCol);
//        if(LogOutput) Log.i(TAG, " RectCorner: " +nextCorner.toString() );
        if(nextCorner[0].x ==0)imageErrer +=2;
        if(nextPoint[0].x ==0)imageErrer +=5;

        //台形の変形率
        double DeformationValue =0.10f ,Deformation
                ,LenLineRow, LenLineCol
                ,prevLenLineSum =0
                ,nextLenLineSum =0;
        double []prevLenLine =new double[4]
                ,nextLenLine =new double[4]
                ,prevLenLinePar =new double[4]
                ,nextLenLinePar =new double[4];
        boolean DeformationOver =false;
        for (int i = 0; i <nextCorner.length ; i++) {
            int Rect_i =i +1;
            if(Rect_i > 3)Rect_i =0;
            LenLineRow =prevCorner[i].y -prevCorner[Rect_i].y;
            LenLineCol =prevCorner[i].x -prevCorner[Rect_i].x;
            prevLenLine[i] =sqrt(pow(LenLineRow,2)+ pow(LenLineCol,2));
            prevLenLineSum +=prevLenLine[i];

            LenLineRow =nextCorner[i].y -nextCorner[Rect_i].y;
            LenLineCol =nextCorner[i].x -nextCorner[Rect_i].x;
            nextLenLine[i] =sqrt(pow(LenLineRow,2)+ pow(LenLineCol,2));
            nextLenLineSum +=nextLenLine[i];
//            if(LogOutput) Log.i(TAG, " nextLenLineSum: " +nextLenLineSum +" prevLenLineSum: " +prevLenLineSum );
        }
        for (int i = 0; i <prevLenLine.length ; i++) {
            prevLenLinePar[i] =prevLenLine[i] /prevLenLineSum;
            nextLenLinePar[i] =nextLenLine[i] /nextLenLineSum;
            Deformation =(prevLenLinePar[i] > nextLenLinePar[i])?
                    prevLenLinePar[i] /nextLenLinePar[i]
                    :nextLenLinePar[i] /prevLenLinePar[i];

//            if(LogOutput) Log.i(TAG, " Deformation: " +Deformation +" DeformationValue: " +DeformationValue );
            if(Deformation -1 <DeformationValue
                    && !Double.isNaN(Deformation))continue;
            DeformationOver=true;
            break;
        }
        if(DeformationOver)imageErrer +=9;

        //処理
        Mat PrevWarp =new Mat()
                ,NextWarp =new Mat()
                ,PrevGaussianBlur =new Mat()
                ,NextGaussianBlur =new Mat()
                ,PrevResize =new Mat()
                ,NextResize =new Mat()
                ,diffMat =new Mat()
                ,CheckMat ,CheckMat2 ;
        Point [] DstPoint=new Point[4];
        Size dstSize =new Size(mPtColCheck *PixelLevel, mPtRowCheck *PixelLevel);
        Pt_i=0;
        DstPoint[Pt_i++] =new Point(0f                     ,0f                     );
        DstPoint[Pt_i++] =new Point(mPtColCheck *PixelLevel,0f                     );
        DstPoint[Pt_i++] =new Point(mPtColCheck *PixelLevel,mPtRowCheck *PixelLevel);
        DstPoint[Pt_i++] =new Point(0f                     ,mPtRowCheck *PixelLevel);

        MatOfPoint2f src_pts =new MatOfPoint2f();
        MatOfPoint2f dst_pts =new MatOfPoint2f();
        src_pts.fromArray(prevCorner);
        dst_pts.fromArray(DstPoint);
        Mat r_mat = Imgproc.getPerspectiveTransform(src_pts, dst_pts);
        Imgproc.warpPerspective(mGray_prev, PrevWarp, r_mat, dstSize);//, Imgproc.INTER_LINEAR);
//        Size kernelSize = new Size(mGaussianBlur,mGaussianBlur);
        if(mGaussianBlur != 1)Imgproc.medianBlur(PrevWarp,PrevGaussianBlur,(int)mGaussianBlur);
//        if(mGaussianBlur != 1)Imgproc.GaussianBlur(PrevWarp,PrevGaussianBlur,kernelSize,mGaussianBlur -1,mGaussianBlur -1);
        else PrevGaussianBlur =PrevWarp;
        Imgproc.resize(PrevGaussianBlur, PrevResize,new Size(mPtColCheck, mPtRowCheck));

        src_pts.fromArray(nextCorner);
        r_mat = Imgproc.getPerspectiveTransform(src_pts, dst_pts);
//        if(LogOutput) Log.i(TAG, " getPerspectiveTransform 2: " );
        Imgproc.warpPerspective(mGray_next, NextWarp, r_mat, dstSize);//, Imgproc.INTER_LINEAR);


        if(mGaussianBlur != 1)Imgproc.medianBlur(NextWarp,NextGaussianBlur,(int)mGaussianBlur);
//        if(mGaussianBlur != 1)Imgproc.GaussianBlur(NextWarp,NextGaussianBlur,kernelSize,mGaussianBlur -1,mGaussianBlur -1);
        else NextGaussianBlur =NextWarp;
        Imgproc.resize(NextGaussianBlur, NextResize,new Size(mPtColCheck, mPtRowCheck));
/*        if(LogOutput) Log.i(TAG, " warpPerspective 2: " );
        if(LogOutput) Log.i(TAG, " imageErrer: " +imageErrer
                +" mGray_prev: " +mGray_prev.toString()
                +" PrevWarp: " +PrevWarp.toString()
                +" PrevGaussianBlur: " +PrevGaussianBlur.toString()
                +" PrevResize: " +PrevResize.toString()
                +" mGray_next: " +mGray_next.toString()
                +" NextWarp: " +NextWarp.toString()
                +" NextGaussianBlur: " +NextGaussianBlur.toString()
                +" NextResize: " +NextResize.toString()
        );*/

        //averageDiff
        double diffSum =0;
        double []diffValue =new double[2];
        Core.absdiff(PrevResize,NextResize,diffMat);
        for (int i = 0; i <PrevResize.rows() ; i++) {
            for (int j = 0; j <PrevResize.cols() ; j++) {
                diffValue =diffMat.get(i,j);
                diffSum +=diffValue[0];
            }
        }
        diffSum /=PrevResize.rows() *PrevResize.cols();
        for (int i = 0; i <PrevResize.rows() ; i++) {
            for (int j = 0; j <PrevResize.cols() ; j++) {
                diffValue =NextResize.get(i,j);
                diffValue[0] -=diffSum;
                NextResize.put(i,j,diffValue);
            }
        }
        Core.absdiff(PrevResize,NextResize,diffMat);

        CheckMat =new Mat(PrevResize.rows(),PrevResize.cols(),PrevResize.type(),new Scalar(0,0,0));
        CheckMat2 =new Mat(PrevResize.rows(),PrevResize.cols(),PrevResize.type(),new Scalar(0,0,0));
        double [] Color, Color_i;
        boolean[][] nextPtAreaCheck   =new boolean[mPtRowCheck][mPtColCheck]
                ,RangeOverCheck =new boolean[mPtRowCheck][mPtColCheck];
        int [][] prevPtDiffCheck =new int[mPtRowCheck][mPtColCheck]
                ,nextPtDiffCheck =new int[mPtRowCheck][mPtColCheck];
        int DiffCheck_i;
        for (int i = 1; i <diffMat.rows()-1 ; i++) {
            for (int j = 1; j < diffMat.cols() - 1; j++) {

                RangeOverCheck[i][j]=false;
                Color = diffMat.get(i, j);
                if (Color[0] < mAlertLevel)
                    RangeOverCheck[i][j]=true;

                DiffCheck_i = 0;
                Color = PrevResize.get(i, j);
                Color_i = PrevResize.get(i + 1, j);
                if (abs(Color[0] - Color_i[0]) > mRangeOverTime)
                    DiffCheck_i += (Color[0] < Color_i[0]) ? 1 : -1;
                Color_i = PrevResize.get(i, j + 1);
                if (abs(Color[0] - Color_i[0]) > mRangeOverTime)
                    DiffCheck_i += (Color[0] < Color_i[0]) ? 2 : -2;
                prevPtDiffCheck[i][j] = DiffCheck_i;

                DiffCheck_i = 0;
                Color = NextResize.get(i, j);
                Color_i = NextResize.get(i + 1, j);
                if (abs(Color[0] - Color_i[0]) > mRangeOverTime)
                    DiffCheck_i += (Color[0] < Color_i[0]) ? 1 : -1;
                Color_i = NextResize.get(i, j + 1);
                if (abs(Color[0] - Color_i[0]) > mRangeOverTime)
                    DiffCheck_i += (Color[0] < Color_i[0]) ? 2 : -2;
                nextPtDiffCheck[i][j] = DiffCheck_i;

                nextPtAreaCheck[i][j] = false;
                if (prevPtDiffCheck[i][j] == nextPtDiffCheck[i][j]
                    //|| prevPtDiffCheck[i][j]!=0
                    //|| nextPtDiffCheck[i][j]!=0
                        ) {
                    nextPtAreaCheck[i][j] = true;
                } //else mSafetyCount--;

                if(!nextPtAreaCheck[i][j] ){
                    Color =CheckMat.get(i, j);
                    Color[0] = 200;
                    CheckMat.put(i, j, Color);
                }
                if( !RangeOverCheck[i][j] ){
                    Color =CheckMat2.get(i, j);
                    Color[0] = 200;
                    CheckMat2.put(i, j, Color);
                }

            }
        }
        Point [] AlertPoint =ReCreatePoint(prevCorner, mPtRowCheck, mPtColCheck);
//        if(LogOutput) Log.i(TAG, " ReCreatePoint: " +AlertPoint);

        boolean[][] AlertPoint4Check =new boolean[mPtRowCheck][mPtColCheck]
                ,AlertPointCheck =new boolean[mPtRowCheck][mPtColCheck];
        for (int i = 0; i <mPtColCheck ; i++) {
            AlertPointCheck[0             ][i             ] =false;
            AlertPointCheck[mPtRowCheck -1][i             ] =false;
        }
        for (int i = 0; i <mPtRowCheck ; i++){
            AlertPointCheck[i             ][0             ] =false;
            AlertPointCheck[i             ][mPtColCheck -1] =false;
        }
        for (int i = 1; i < mPtRowCheck -1 ; i++) {
            for (int j = 1; j < mPtColCheck -1; j++) {
                int PtXY =i *mPtColCheck +j;
                AlertPointCheck[i][j] =true;

                if( nextPtAreaCheck[i][j] && CheckOff !=3)AlertPointCheck[i][j]=false;
                if(  RangeOverCheck[i][j] && CheckOff !=2)AlertPointCheck[i][j]=false;

                //AlertPoint4Check
                AlertPoint4Check[i][j] =false;
                if( AlertPointCheck[i][j]){
                    int AlertCount4 =2 ;
                    if(AlertPointCheck[i   ][j -1])AlertCount4--;
                    if(AlertPointCheck[i -1][j   ])AlertCount4--;
                    if(AlertPointCheck[i -1][j -1])AlertCount4--;
                    if(AlertPointCheck[i   ][j +1])AlertCount4--;
                    if(AlertPointCheck[i +1][j   ])AlertCount4--;
                    if(AlertPointCheck[i +1][j +1])AlertCount4--;
                    if(AlertPointCheck[i -1][j +1])AlertCount4--;
                    if(AlertPointCheck[i +1][j -1])AlertCount4--;
                    if(AlertCount4 <= 0)AlertPoint4Check[i][j] =true;
                }

                Pt_i =i /(mPtRowCheck /AlertArea.length +1) ;
                if(mPtAngle ==1)Pt_i =(AlertArea.length -1) -j /(mPtColCheck /AlertArea.length +1) ;
                if(mPtAngle ==2)Pt_i =(AlertArea.length -1) -i /(mPtRowCheck /AlertArea.length +1) ;
                if(mPtAngle ==3)Pt_i =j /(mPtCol /AlertArea.length +1) ;
                //PointClare
                //if(LogOutput)
                if(!AlertPoint4Check[i][j] && CheckOff !=1){
                    AlertPoint[PtXY].x =0;
                    AlertPoint[PtXY].y =0;
                    mSafetyCount--;
                }else{
                    AlertArea[Pt_i]++;
                    mErrerLevel--;
                }
            }
        }
//        if(LogOutput) Log.i(TAG, " AlertPointCheck: " +AlertPointCheck);
        //WalkCheck**
        boolean mWalkStop =false;
        double WalkLine =0,MoveRow,MoveCol;
        int WalkPoint =4;
        Point MoveCenter =new Point();
        for (int i = 0; i <prevCorner.length ; i++) {
            MoveRow =nextCorner[i].y -prevCorner[i].y;
            MoveCol =nextCorner[i].x -prevCorner[i].x;
            MoveCenter.x +=MoveCol;
            MoveCenter.y +=MoveRow;
            WalkLine +=sqrt(pow(MoveRow,2)+ pow(MoveCol,2));
            if(WalkLine >= mWalkLevel)WalkPoint--;
        }
        if(WalkPoint >0)mWalkStop =true;
        MoveCenter.x /=4;
        MoveCenter.x +=center.x;
        MoveCenter.y /=4;
        MoveCenter.y +=center.y;
//        if(LogOutput) Log.i(TAG, " WalkCheck: " +mWalkStop);

        // オプティカルフローの表示

        int CheckImageCount =5;
        int ProcessAreaTime =3;
        mGray_Draw =new Mat(mGray_prev.rows() +mPtRowCheck *ProcessAreaTime,mGray_prev.cols() ,mGray_prev.type());
//        if(LogOutput) Log.i(TAG, " mGray_Draw: " +mGray_Draw.toString());
        Mat Process_Draw =new Mat(mPtRowCheck,mGray_prev.cols(),mGray_prev.type());
//        if(LogOutput) Log.i(TAG, " Process_Draw: " +Process_Draw.toString());


        if(BackGroundView >0) {
//            if(LogOutput) Log.i(TAG, " Gray_Draw Value: " +" mPtRowCheck: " +mPtRowCheck +" mPtColCheck: " +mPtColCheck);
//            if(LogOutput) Log.i(TAG, " Gray_Draw Size 0: " +(mGray_prev.cols()/ProcessAreaTime -mPtColCheck *CheckImageCount *ProcessAreaTime)
//                    +" Size 1: " +(mGray_prev.rows()/ProcessAreaTime -mPtColCheck *CheckImageCount *ProcessAreaTime));
            if((mGray_prev.rows()/ProcessAreaTime -mPtColCheck *CheckImageCount *ProcessAreaTime) <0)
                ProcessAreaTime =2;
            Mat Gray_Draw;
            if(mPtAngle %2 ==0)Gray_Draw =new Mat(mPtRowCheck ,mGray_prev.cols()/ProcessAreaTime -mPtColCheck *CheckImageCount *ProcessAreaTime ,mGray_prev.type(),new Scalar(100,100,100));
            else Gray_Draw =new Mat(mPtRowCheck ,mGray_prev.rows()/ProcessAreaTime -mPtColCheck *CheckImageCount *ProcessAreaTime ,mGray_prev.type(),new Scalar(100,100,100));
//            if(LogOutput) Log.i(TAG, " Gray_Draw: " +Gray_Draw.toString());
            //処理画像連結
            List <Mat> srcMat;
/*            if(LogOutput) Log.i(TAG, " PrevWarp: " +PrevWarp.toString()
                    +" NextWarp: " +NextWarp.toString()
                    +" Gray_Draw: " +Gray_Draw.toString());*/
            srcMat = new ArrayList<Mat>();
            srcMat.add(PrevResize);
            srcMat.add(NextResize);
            srcMat.add(diffMat);
            srcMat.add(CheckMat);
            srcMat.add(CheckMat2);
            srcMat.add(Gray_Draw);
            Core.hconcat(srcMat,Process_Draw);
            Imgproc.resize(Process_Draw, Process_Draw,
                    new Size(mGray_prev.cols(), mPtRowCheck *ProcessAreaTime));
//            if(LogOutput) Log.i(TAG, " Process_Draw: " +Process_Draw.toString());

            //原画像連結
            srcMat = new ArrayList<Mat>();
            srcMat.add(mGray_prev);
            srcMat.add(Process_Draw);
            Core.vconcat(srcMat, mGray_Draw);
//            if(LogOutput) Log.i(TAG, " mGray_Draw: " + mGray_Draw.toString());
            Imgproc.rectangle(mGray_Draw, new Point(0, 0), new Point(mGray_prev.cols() - 3, mGray_prev.rows() - 3), new Scalar(250, 250, 250), 3);
            Imgproc.rectangle(mGray_Draw, new Point(0, mGray_prev.rows() - 3), new Point(mGray_Draw.cols() - 3, mGray_Draw.rows() - 3), new Scalar(250, 250, 250), 3);
        }else {
            mGray_Draw =mGray_prev.clone();
        }

        mGray_prev.release();
        if(mCameraAngle != 0)// mGray_Draw =mGray_prev.clone();
            mGray_Draw =rotationAffine(mGray_Draw, mCameraAngle);

        for (int i =0 ; i < nextCorner.length ; i++){
            int Rect_i =i +1;
            if(Rect_i > 3)Rect_i =0;
            if(nextCorner[0].x !=0){
                Imgproc.line(mGray_Draw, PointRotate( prevCorner[i],mCameraAngle , mGray_Draw)
                        ,PointRotate( prevCorner[Rect_i],mCameraAngle , mGray_Draw),new Scalar(250,250,250),2);
                Imgproc.line(mGray_Draw, PointRotate( AlertRect[i],mCameraAngle , mGray_Draw)
                        ,PointRotate( AlertRect[Rect_i],mCameraAngle , mGray_Draw),new Scalar(250,250,250),6);

                if(BackGroundView >0)
                    Imgproc.line(mGray_Draw, PointRotate( nextCorner[i],mCameraAngle , mGray_Draw)
                            ,PointRotate( nextCorner[Rect_i],mCameraAngle , mGray_Draw),new Scalar(250,250,250),1);
            }
        }
        if(imageErrer <0)
            for (int i = 0; i <AlertPoint.length ; i++) {
                Imgproc.circle(mGray_Draw,PointRotate( AlertPoint[i],mCameraAngle , mGray_Draw),5,new Scalar(250,250,250),-1);
            }

        Imgproc.circle(mGray_Draw,PointRotate( center,mCameraAngle , mGray_Draw),(int) mWalkLevel,new Scalar(250,250,250),2);
        Imgproc.circle(mGray_Draw,PointRotate( center,mCameraAngle , mGray_Draw), searchSize,new Scalar(250,250,250),2);
        Imgproc.line(mGray_Draw, PointRotate( new Point(MoveCenter.x -(double) 10,MoveCenter.y),mCameraAngle , mGray_Draw)
                ,PointRotate( new Point(MoveCenter.x +(double) 10,MoveCenter.y),mCameraAngle , mGray_Draw),new Scalar(250,250,250),2);
        Imgproc.line(mGray_Draw, PointRotate( new Point(MoveCenter.x ,MoveCenter.y-(double) 10),mCameraAngle , mGray_Draw)
                ,PointRotate( new Point(MoveCenter.x ,MoveCenter.y +(double) 10),mCameraAngle , mGray_Draw),new Scalar(250,250,250),2);

        //AlertDraw
        final int AlertDrawPoint =5;
        int AlertDrawPitch       =mGray_Draw.cols()/AlertDrawPoint;
        int RectWidth            =10;
        for (int i =0; i < AlertDrawPoint ; i++) {
            Point mRectSt= new Point(i *AlertDrawPitch, 0);
            Point mRectEd= new Point((i +1) *AlertDrawPitch -RectWidth, RectWidth);
            if(        (i == 0 && mWalkStop)
                    || (i == 1 && mSafetyCount < 0 && mErrerLevel >0)
                    || (i >= 2 && AlertArea[i -2] > mAlertAreaCount))
                Imgproc.rectangle(mGray_Draw, mRectSt, mRectEd ,new Scalar(250,250,250),10);
        }

/*        if (LogOutput)
            Log.d(TAG, " AlertDraw:"
                    + " mWalkStop:" + mWalkStop
                    + " mSafetyCount:" + mSafetyCount
                    + " mErrerLevel:" + mErrerLevel
                    + " imageErrer:" + imageErrer
                    + " AlertArea 0:" + AlertArea[0]
                    + " AlertArea 1:" + AlertArea[1]
                    + " AlertArea 2:" + AlertArea[2]
            );*/
        int NoticeSound =-1;
        if(        !mWalkStop
                && mSafetyCount < 0
                && mErrerLevel  > 0
                && imageErrer   < 0){
            NoticeSound =0;
            if(AlertArea[0] > mAlertAreaCount)NoticeSound =3;
            if(AlertArea[1] > mAlertAreaCount)NoticeSound =3;
            if(AlertArea[2] > mAlertAreaCount){
                mAlertCount--;
                if(mAlertCount <=0)
                    NoticeSound =4;
            }else
                mAlertCount =(int)AlertCount;
        }
        //SoundAlert
        if(NoticeSound >= 0)
            if(Vibration == 0) {
            // play(ロードしたID, 左音量, 右音量, 優先度, ループ,再生速度)
            soundPool.play(soundList[NoticeSound], 1.0f, 1.0f, 0, 0, 1);
            }else {
                Vibrator mVibrator =(Vibrator)getSystemService(VIBRATOR_SERVICE);
                if(NoticeSound == 4) mVibrator.vibrate(500);
                else mVibrator.vibrate(20);
            }

        if(BackGroundView ==0 && NoticeSound == 4){
            int AlertSize =(mGray_Draw.cols() < mGray_Draw.rows())?
                    mGray_Draw.cols()/3 : mGray_Draw.rows()/3 ;
            Imgproc.circle(mGray_Draw,PointRotate( center,mCameraAngle , mGray_Draw), AlertSize,new Scalar(250,250,250),20);
        }
        Calendar mCd =Calendar.getInstance();
        if(Millis == 0) Millis =mCd.getTimeInMillis();
        else{
            long mMillis =mCd.getTimeInMillis();
            long Millis_i = 1000 /2 -(mMillis -Millis);
            if(Millis_i < 0)Millis_i =0;
            try {
                Thread.sleep(Millis_i);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            Millis =mMillis;
        }
        mGray_prev.release();
        mGray_next.release();
        return mGray_Draw;
    }

    private long Millis =0;
    int mAlertCount =0;

    private Mat rotationAffine(Mat src, double angle){
        // assuming source image's with and height are a pair value:
        //Image is rotated - cropped-to-fit dst Mat.
        int centerX = Math.round(src.width()/2);
        int centerY = Math.round(src.height()/2);

        Point center = new Point(centerX,centerY);
        double scale = 1.0f;
        Mat mapMatrix = Imgproc.getRotationMatrix2D(center, angle, scale);
        Mat mIntermediateMat;
        if(angle == 0 || angle ==180 || angle == -180 || angle == 360) {
            mIntermediateMat = new Mat(src.rows(), src.cols(), src.type());
        } else {
            int IntermediateSize = src.height()> src.width() ? src.height(): src.width();
            center = new Point(IntermediateSize/2 ,IntermediateSize/2);
            mapMatrix = Imgproc.getRotationMatrix2D(center, angle, scale);
            mIntermediateMat =new Mat(IntermediateSize,IntermediateSize,src.type());
        }
        Imgproc.warpAffine(src, mIntermediateMat, mapMatrix, mIntermediateMat.size(), INTER_LINEAR);
        if(angle == 90.0f)mIntermediateMat =mIntermediateMat.submat(0, mIntermediateMat. rows(), 0, src.rows());
        if(angle == 270.0f)mIntermediateMat =mIntermediateMat.submat(0, mIntermediateMat.rows(), mIntermediateMat.cols() -src.rows(), mIntermediateMat.cols());

        return mIntermediateMat;
    }

    private Point[] RectCorner(Point[] Point,int mPtRow ,int mPtCol){
        int [][] minPtY =new int [mPtRow][mPtCol]
                ,maxPtY =new int[mPtRow][mPtCol]
                ,CountPtY =new int[mPtRow][mPtCol]
                ,minPtX =new int [mPtRow][mPtCol]
                ,maxPtX =new int[mPtRow][mPtCol]
                ,CountPtX =new int[mPtRow][mPtCol];
        double [][] ThetaPtY =new double[mPtRow][mPtCol]
                ,ThetaPtX =new double[mPtRow][mPtCol];

        float Theta_i,PointY ,PointX;
        Point [] nextCorner =new Point[4];

        float RectCornerAngleLevel =2.0f;
        int P_i ,maxCount, CornerCount = 4;
        int [][] CandidatesXY =new int [4][3]; //0:min 1:Max 2:i;
        int [] Point_XY =new int[2];

        for (int i = 0; i <maxPtY.length ; i++)
            for (int j = 0; j <maxPtY[i].length ; j++) {
                maxPtY[i][j] = -1;
                maxPtX[i][j] = -1;
            }
        //x
        for (int i = 0; i <mPtRow ; i++) {
            for (int j = 0; j <mPtCol-2 ; j++) {
                if (maxPtX[i][j] >= 0) continue;
//                Point_XY[0] = i +mPtCol *j;
//                Point_XY[1] = i +mPtCol *j +mPtCol;
                Point_XY[0] = i * mPtCol + j;
                Point_XY[1] = i * mPtCol + j +1;

                PointY = (float) (Point[Point_XY[1]].y - Point[Point_XY[0]].y);
                PointX = (float) (Point[Point_XY[1]].x - Point[Point_XY[0]].x);
                ThetaPtX[i][j] = fastAtan2(PointY, PointX);
                minPtX[i][j] = Point_XY[0];
                CountPtX[i][j] = 0;
                for (int k = j +2; k < mPtCol; k++) {
                    if (maxPtX[i][k] >= 0) continue;
//                    Point_XY[1] = i +mPtCol *k;
                    Point_XY[1] = i *mPtCol +k;
//                    Log.i(TAG, " RectCorner x: " +" i:" +i +" j:" +j +" k:" +k +" Point_XY[1]:" +Point_XY[1] );
                    PointY = (float) (Point[Point_XY[1]].y - Point[Point_XY[0]].y);
                    PointX = (float) (Point[Point_XY[1]].x - Point[Point_XY[0]].x);
                    Theta_i = fastAtan2(PointY, PointX);
                    Theta_i -= ThetaPtX[i][j];
                    if(Theta_i < -180)Theta_i +=360;
                    if(Theta_i >= 180)Theta_i -=360;
                    if (-RectCornerAngleLevel > Theta_i
                            || Theta_i > RectCornerAngleLevel) continue;
                    maxPtX[i][j] = Point_XY[1];
                    maxPtX[i][k] = 0;
                    CountPtX[i][j]++;
                }
            }
        }

//        Log.i(TAG, " RectCorner x: CheckRun");
        for (int i = 0; i <CountPtX.length /3 ; i++) {
            maxCount=0;
            P_i =0;
            for (int j = 0; j <CountPtX[i].length -2 ; j++) {
                if (maxCount < CountPtX[i][j]) {
                    maxCount = CountPtX[i][j];
                    P_i = j;
//                    Log.i(TAG, " RectCorner x: " +" maxCount:" +maxCount );
                }
            }
            if(maxCount >mPtCol/2){
                CandidatesXY[0][0]=minPtX[i][P_i];
                CandidatesXY[0][1]=maxPtX[i][P_i];
                CandidatesXY[0][2]=i;
                CornerCount--;
                break;
            }
        }
//        if(LogOutput)
//            Log.i(TAG, " RectCorner x: " +" CornerCount:" +CornerCount );
        for (int i = CountPtX.length -1; i >CountPtX.length /3 ; i--) {
            maxCount=0;
            P_i =0;
            for (int j = 0; j <CountPtX[i].length -2 ; j++) {
                if (maxCount < CountPtX[i][j]) {
                    maxCount = CountPtX[i][j];
                    P_i = j;
                }
            }
            if(maxCount >mPtCol /2){
                CandidatesXY[2][0]=minPtX[i][P_i];
                CandidatesXY[2][1]=maxPtX[i][P_i];
                CandidatesXY[2][2]=i;
                CornerCount--;
                break;
            }
        }
//        if(LogOutput)
//            Log.i(TAG, " RectCorner x: " +" CornerCount:" +CornerCount );
        //y
        for (int i = 0; i <mPtRow ; i++) {
            for (int j = 0; j <mPtCol -2 ; j++) {
                if (maxPtY[i][j] >= 0) continue;
//                Point_XY[0] = i * mPtCol + j;
//                Point_XY[1] = i * mPtCol + j +1;
                Point_XY[0] = i +mPtCol *j;
                Point_XY[1] = i +mPtCol *j +mPtCol;

                PointY = (float) (Point[Point_XY[1]].y - Point[Point_XY[0]].y);
                PointX = (float) (Point[Point_XY[1]].x - Point[Point_XY[0]].x);
                ThetaPtY[i][j] = fastAtan2(PointY, PointX);
                minPtY[i][j] = Point_XY[0];
                CountPtY[i][j] = 0;
                for (int k = j + 2; k < mPtCol; k++) {
                    if (maxPtY[i][k] >= 0) continue;
//                    Log.i(TAG, " RectCorner y: " +" i:" +i +" j:" +j +" k:" +k +" Point_XY[1]:" +Point_XY[1] );
//                    Point_XY[1] = i *mPtCol +k;
                    Point_XY[1] = i +mPtCol *k;
                    PointY = (float) (Point[Point_XY[1]].y - Point[Point_XY[0]].y);
                    PointX = (float) (Point[Point_XY[1]].x - Point[Point_XY[0]].x);
                    Theta_i = fastAtan2(PointY, PointX);
                    Theta_i -= ThetaPtY[i][j];
                    if(Theta_i < -180)Theta_i +=360;
                    if(Theta_i >= 180)Theta_i -=360;
                    if (-RectCornerAngleLevel > Theta_i
                            || Theta_i > RectCornerAngleLevel) continue;
                    maxPtY[i][j] = Point_XY[1];
                    maxPtY[i][k] = 0;
                    CountPtY[i][j]++;
                }
            }
        }

        for (int i = 0; i <CountPtY.length /3 ; i++) {
            maxCount=0;
            P_i =0;
            for (int j = 0; j <CountPtY[i].length -2 ; j++) {
                if (maxCount < CountPtY[i][j]) {
                    maxCount = CountPtY[i][j];
                    P_i = j;
                }
            }
            if(maxCount >mPtCol /2){
                CandidatesXY[1][0]=minPtY[i][P_i];
                CandidatesXY[1][1]=maxPtY[i][P_i];
                CandidatesXY[1][2]=i;
                CornerCount--;
                break;
            }
        }
//        if(LogOutput)
//            Log.i(TAG, " RectCorner y: " +" CornerCount:" +CornerCount );
        for (int i = CountPtY.length -1; i >CountPtY.length /3 ; i--) {
            maxCount=0;
            P_i =0;
            for (int j = 0; j <CountPtY[i].length -2 ; j++) {
                if (maxCount < CountPtY[i][j]) {
                    maxCount = CountPtY[i][j];
                    P_i = j;
                }
            }
            if(maxCount >mPtCol/2){
                CandidatesXY[3][0]=minPtY[i][P_i];
                CandidatesXY[3][1]=maxPtY[i][P_i];
                CandidatesXY[3][2]=i;
                CornerCount--;
                break;
            }
        }
//        if(LogOutput)
//            Log.i(TAG, " RectCorner y: " +" CornerCount:" +CornerCount );

        double [] prevP_y =new double[4];
        double [] prevP_x =new double[4];
        for (int k = 0; k < prevP_y.length ; k++) {
            P_i =k +1;
            if(P_i > 3) P_i =0;
            prevP_y[0  ] =Point[CandidatesXY[k  ][0]].y;
            prevP_x[0  ] =Point[CandidatesXY[k  ][0]].x;
            prevP_y[2  ] =Point[CandidatesXY[k  ][1]].y;
            prevP_x[2  ] =Point[CandidatesXY[k  ][1]].x;
            prevP_y[1  ] =Point[CandidatesXY[P_i][0]].y;
            prevP_x[1  ] =Point[CandidatesXY[P_i][0]].x;
            prevP_y[3  ] =Point[CandidatesXY[P_i][1]].y;
            prevP_x[3  ] =Point[CandidatesXY[P_i][1]].x;


            double S1 = ((prevP_x[3] -prevP_x[1])  *(prevP_y[0] -prevP_y[1]) -(prevP_y[3] -prevP_y[1]) *(prevP_x[0] -prevP_x[1])) /2;
            double S2 = ((prevP_x[3] -prevP_x[1])  *(prevP_y[1] -prevP_y[2]) -(prevP_y[3] -prevP_y[1]) *(prevP_x[1] -prevP_x[2])) /2;
            double C1_X = prevP_x[0] +(prevP_x[2]  -prevP_x[0]) *S1 /(S1 +S2);
            double C1_Y = prevP_y[0] +(prevP_y[2]  -prevP_y[0]) *S1 /(S1 +S2);
            nextCorner[k] =new Point(C1_X,C1_Y);
        }
/*        if(LogOutput)
            for (int i = 0; i <CandidatesXY.length ; i++) {
                Log.i(TAG, " RectCorner xy: "
                        +" i:" +i
                        +" CandidatesXY 0:" +CandidatesXY[i  ][0]
                        +" CandidatesXY 1:" +CandidatesXY[i  ][1]
                        +" CandidatesXY 1:" +CandidatesXY[i  ][2]
                        +" min:" +Point[CandidatesXY[i  ][0]]
                        +" max:" +Point[CandidatesXY[i  ][1]]
                        +" nextCorner:" +nextCorner[i  ]
                        +" Point:" +Point[i  ]
                );
            }*/
        //補正
        for (int i = 0; i <nextCorner.length ; i++) {
            if( i ==0 && CandidatesXY[i][2] == 0)continue;
            if( i ==1 && CandidatesXY[i][2] == 0)continue;
            if( i ==2 && CandidatesXY[i][2] == mPtCol -1)continue;
            if( i ==3 && CandidatesXY[i][2] == mPtRow -1)continue;
//            if(LogOutput)continue;
            double wide_Len, wide_i;

            if(i == 0){//x:0-3
                wide_i =CandidatesXY[2][2] -CandidatesXY[0][2] ;
                wide_Len =nextCorner[1].x -nextCorner[0].x;
                wide_Len /=wide_i ;
                wide_Len *=(double) CandidatesXY[0][2] ;
                nextCorner[0].x -=wide_Len;
                wide_Len =nextCorner[3].x -nextCorner[2].x;
                wide_Len /=wide_i;
                wide_Len *=(double) CandidatesXY[0][2] ;
                nextCorner[3].x +=wide_Len;
            }
            if(i == 2){//x:1-2
                wide_i =CandidatesXY[2][2] -CandidatesXY[0][2] ;
                wide_Len =nextCorner[1].x -nextCorner[0].x;
                wide_Len /=wide_i;
                wide_Len *=(double) (mPtCol -CandidatesXY[2][2] -1);
                nextCorner[1].x +=wide_Len;
                wide_Len =nextCorner[3].x -nextCorner[2].x;
                wide_Len /=wide_i;
                wide_Len *=(double) (mPtCol -CandidatesXY[2][2] -1);
                nextCorner[2].x -=wide_Len;
            }
            if(i == 1){//y:0-1
                wide_i =CandidatesXY[3][2] -CandidatesXY[1][2] ;
                wide_Len =nextCorner[3].y -nextCorner[0].y;
                wide_Len /=wide_i;
                wide_Len *=(double) CandidatesXY[1][2] ;
                nextCorner[0].y -=wide_Len;
                wide_Len =nextCorner[2].y -nextCorner[1].y;
                wide_Len /=wide_i;
                wide_Len *=(double) CandidatesXY[1][2] ;
                nextCorner[1].y -=wide_Len;
            }
            if(i == 3){//x:3-2
                wide_i =CandidatesXY[3][2] -CandidatesXY[1][2] ;
                wide_Len =nextCorner[3].y -nextCorner[0].y;
                wide_Len /=wide_i;
                wide_Len *=(double) (mPtRow -CandidatesXY[3][2] -1);
                nextCorner[3].y +=wide_Len;
                wide_Len =nextCorner[2].y -nextCorner[1].y;
                wide_Len /=wide_i;
                wide_Len *=(double) (mPtRow -CandidatesXY[3][2] -1);
                nextCorner[2].y -=wide_Len;
            }
        }
/*        if(LogOutput)
            for (int i = 0; i <CandidatesXY.length ; i++) {
                Log.i(TAG, " RectCornerCandidates xy: "
                        +" i:" +i
                        +" nextCorner:" +nextCorner[i]
                );
            }*/
        if(CornerCount >0){
            for (int i = 0; i <nextCorner.length ; i++) {
                nextCorner[i].y =0;
                nextCorner[i].x =0;
            }
        }
        return nextCorner;
    }
    private Point[] RectCorner(Point[] Point, MatOfByte Status,int mPtRow ,int mPtCol){
        int [][] minPtY =new int [mPtRow][mPtCol]
                ,maxPtY =new int[mPtRow][mPtCol]
                ,CountPtY =new int[mPtRow][mPtCol]
                ,minPtX =new int [mPtRow][mPtCol]
                ,maxPtX =new int[mPtRow][mPtCol]
                ,CountPtX =new int[mPtRow][mPtCol];
        double [][] ThetaPtY =new double[mPtRow][mPtCol]
                ,ThetaPtX =new double[mPtRow][mPtCol];

        float Theta_i,PointY ,PointX;
        Point [] nextCorner =new Point[4];

        byte[] StatusByte =Status.toArray();

        float RectCornerAngleLevel =2.0f;
        int P_i ,maxCount, CornerCount = 4;
        int [][] CandidatesXY =new int [4][3]; //0:min 1:Max 2:i;
        int [] Point_XY =new int[2];

        for (int i = 0; i <maxPtY.length ; i++)
            for (int j = 0; j <maxPtY[i].length ; j++) {
                maxPtY[i][j] = -1;
                maxPtX[i][j] = -1;
            }
        //x
        for (int i = 0; i <mPtRow ; i++) {
            for (int j = 0; j <mPtCol-2 ; j++) {
                if (maxPtX[i][j] >= 0) continue;
//                Point_XY[0] = i +mPtCol *j;
//                Point_XY[1] = i +mPtCol *j +mPtCol;
                Point_XY[0] = i * mPtCol + j;
                Point_XY[1] = i * mPtCol + j +1;

                if(StatusByte[Point_XY[0]] <0
                        ||StatusByte[Point_XY[1]] <0)
                    continue;
                PointY = (float) (Point[Point_XY[1]].y - Point[Point_XY[0]].y);
                PointX = (float) (Point[Point_XY[1]].x - Point[Point_XY[0]].x);
                ThetaPtX[i][j] = fastAtan2(PointY, PointX);
                minPtX[i][j] = Point_XY[0];
                CountPtX[i][j] = 0;
                for (int k = j +2; k < mPtCol; k++) {
                    if (maxPtX[i][k] >= 0) continue;
//                    Point_XY[1] = i +mPtCol *k;
                    Point_XY[1] = i *mPtCol +k;
//                    Log.i(TAG, " RectCorner x: " +" i:" +i +" j:" +j +" k:" +k +" Point_XY[1]:" +Point_XY[1] );
                    PointY = (float) (Point[Point_XY[1]].y - Point[Point_XY[0]].y);
                    PointX = (float) (Point[Point_XY[1]].x - Point[Point_XY[0]].x);
                    Theta_i = fastAtan2(PointY, PointX);
                    Theta_i -= ThetaPtX[i][j];
                    if(Theta_i < -180)Theta_i +=360;
                    if(Theta_i >= 180)Theta_i -=360;
                    if (-RectCornerAngleLevel > Theta_i
                            || Theta_i > RectCornerAngleLevel) continue;
                    maxPtX[i][j] = Point_XY[1];
                    maxPtX[i][k] = 0;
                    CountPtX[i][j]++;
                }
            }
        }

//        Log.i(TAG, " RectCorner x: CheckRun");
        for (int i = 0; i <CountPtX.length /3 ; i++) {
            maxCount=0;
            P_i =0;
            for (int j = 0; j <CountPtX[i].length -2 ; j++) {
                if (maxCount < CountPtX[i][j]) {
                    maxCount = CountPtX[i][j];
                    P_i = j;
//                    Log.i(TAG, " RectCorner x: " +" maxCount:" +maxCount );
                }
            }
            if(maxCount >mPtCol/2){
                CandidatesXY[0][0]=minPtX[i][P_i];
                CandidatesXY[0][1]=maxPtX[i][P_i];
                CandidatesXY[0][2]=i;
                CornerCount--;
                break;
            }
        }
//        if(LogOutput)
//            Log.i(TAG, " RectCorner x: " +" CornerCount:" +CornerCount );
        for (int i = CountPtX.length -1; i >CountPtX.length /3 ; i--) {
            maxCount=0;
            P_i =0;
            for (int j = 0; j <CountPtX[i].length -2 ; j++) {
                if (maxCount < CountPtX[i][j]) {
                    maxCount = CountPtX[i][j];
                    P_i = j;
                }
            }
            if(maxCount >mPtCol /2){
                CandidatesXY[2][0]=minPtX[i][P_i];
                CandidatesXY[2][1]=maxPtX[i][P_i];
                CandidatesXY[2][2]=i;
                CornerCount--;
                break;
            }
        }
//        if(LogOutput)
//            Log.i(TAG, " RectCorner x: " +" CornerCount:" +CornerCount );
        //y
        for (int i = 0; i <mPtRow ; i++) {
            for (int j = 0; j <mPtCol -2 ; j++) {
                if (maxPtY[i][j] >= 0) continue;
//                Point_XY[0] = i * mPtCol + j;
//                Point_XY[1] = i * mPtCol + j +1;
                Point_XY[0] = i +mPtCol *j;
                Point_XY[1] = i +mPtCol *j +mPtCol;

                PointY = (float) (Point[Point_XY[1]].y - Point[Point_XY[0]].y);
                PointX = (float) (Point[Point_XY[1]].x - Point[Point_XY[0]].x);
                ThetaPtY[i][j] = fastAtan2(PointY, PointX);
                minPtY[i][j] = Point_XY[0];
                CountPtY[i][j] = 0;
                for (int k = j + 2; k < mPtCol; k++) {
                    if (maxPtY[i][k] >= 0) continue;
//                    Log.i(TAG, " RectCorner y: " +" i:" +i +" j:" +j +" k:" +k +" Point_XY[1]:" +Point_XY[1] );
//                    Point_XY[1] = i *mPtCol +k;
                    Point_XY[1] = i +mPtCol *k;
                    PointY = (float) (Point[Point_XY[1]].y - Point[Point_XY[0]].y);
                    PointX = (float) (Point[Point_XY[1]].x - Point[Point_XY[0]].x);
                    Theta_i = fastAtan2(PointY, PointX);
                    Theta_i -= ThetaPtY[i][j];
                    if(Theta_i < -180)Theta_i +=360;
                    if(Theta_i >= 180)Theta_i -=360;
                    if (-RectCornerAngleLevel > Theta_i
                            || Theta_i > RectCornerAngleLevel) continue;
                    maxPtY[i][j] = Point_XY[1];
                    maxPtY[i][k] = 0;
                    CountPtY[i][j]++;
                }
            }
        }

        for (int i = 0; i <CountPtY.length /3 ; i++) {
            maxCount=0;
            P_i =0;
            for (int j = 0; j <CountPtY[i].length -2 ; j++) {
                if (maxCount < CountPtY[i][j]) {
                    maxCount = CountPtY[i][j];
                    P_i = j;
                }
            }
            if(maxCount >mPtCol /2){
                CandidatesXY[1][0]=minPtY[i][P_i];
                CandidatesXY[1][1]=maxPtY[i][P_i];
                CandidatesXY[1][2]=i;
                CornerCount--;
                break;
            }
        }
//        if(LogOutput)
//            Log.i(TAG, " RectCorner y: " +" CornerCount:" +CornerCount );
        for (int i = CountPtY.length -1; i >CountPtY.length /3 ; i--) {
            maxCount=0;
            P_i =0;
            for (int j = 0; j <CountPtY[i].length -2 ; j++) {
                if (maxCount < CountPtY[i][j]) {
                    maxCount = CountPtY[i][j];
                    P_i = j;
                }
            }
            if(maxCount >mPtCol/2){
                CandidatesXY[3][0]=minPtY[i][P_i];
                CandidatesXY[3][1]=maxPtY[i][P_i];
                CandidatesXY[3][2]=i;
                CornerCount--;
                break;
            }
        }
//        if(LogOutput)
//            Log.i(TAG, " RectCorner y: " +" CornerCount:" +CornerCount );

        double [] prevP_y =new double[4];
        double [] prevP_x =new double[4];
        for (int k = 0; k < prevP_y.length ; k++) {
            P_i =k +1;
            if(P_i > 3) P_i =0;
            prevP_y[0  ] =Point[CandidatesXY[k  ][0]].y;
            prevP_x[0  ] =Point[CandidatesXY[k  ][0]].x;
            prevP_y[2  ] =Point[CandidatesXY[k  ][1]].y;
            prevP_x[2  ] =Point[CandidatesXY[k  ][1]].x;
            prevP_y[1  ] =Point[CandidatesXY[P_i][0]].y;
            prevP_x[1  ] =Point[CandidatesXY[P_i][0]].x;
            prevP_y[3  ] =Point[CandidatesXY[P_i][1]].y;
            prevP_x[3  ] =Point[CandidatesXY[P_i][1]].x;


            double S1 = ((prevP_x[3] -prevP_x[1])  *(prevP_y[0] -prevP_y[1]) -(prevP_y[3] -prevP_y[1]) *(prevP_x[0] -prevP_x[1])) /2;
            double S2 = ((prevP_x[3] -prevP_x[1])  *(prevP_y[1] -prevP_y[2]) -(prevP_y[3] -prevP_y[1]) *(prevP_x[1] -prevP_x[2])) /2;
            double C1_X = prevP_x[0] +(prevP_x[2]  -prevP_x[0]) *S1 /(S1 +S2);
            double C1_Y = prevP_y[0] +(prevP_y[2]  -prevP_y[0]) *S1 /(S1 +S2);
            nextCorner[k] =new Point(C1_X,C1_Y);
        }
/*        if(LogOutput)
            for (int i = 0; i <CandidatesXY.length ; i++) {
                Log.i(TAG, " RectCorner xy: "
                        +" i:" +i
                        +" CandidatesXY 0:" +CandidatesXY[i  ][0]
                        +" CandidatesXY 1:" +CandidatesXY[i  ][1]
                        +" CandidatesXY 1:" +CandidatesXY[i  ][2]
                        +" min:" +Point[CandidatesXY[i  ][0]]
                        +" max:" +Point[CandidatesXY[i  ][1]]
                        +" nextCorner:" +nextCorner[i  ]
                        +" Point:" +Point[i  ]
                );
            }*/
        //補正
        for (int i = 0; i <nextCorner.length ; i++) {
            if( i ==0 && CandidatesXY[i][2] == 0)continue;
            if( i ==1 && CandidatesXY[i][2] == 0)continue;
            if( i ==2 && CandidatesXY[i][2] == mPtCol -1)continue;
            if( i ==3 && CandidatesXY[i][2] == mPtRow -1)continue;
//            if(LogOutput)continue;
            double wide_Len, wide_i;

            if(i == 0){//x:0-3
                wide_i =CandidatesXY[2][2] -CandidatesXY[0][2] ;
                wide_Len =nextCorner[1].x -nextCorner[0].x;
                wide_Len /=wide_i ;
                wide_Len *=(double) CandidatesXY[0][2] ;
                nextCorner[0].x -=wide_Len;
                wide_Len =nextCorner[3].x -nextCorner[2].x;
                wide_Len /=wide_i;
                wide_Len *=(double) CandidatesXY[0][2] ;
                nextCorner[3].x +=wide_Len;
            }
            if(i == 2){//x:1-2
                wide_i =CandidatesXY[2][2] -CandidatesXY[0][2] ;
                wide_Len =nextCorner[1].x -nextCorner[0].x;
                wide_Len /=wide_i;
                wide_Len *=(double) (mPtCol -CandidatesXY[2][2] -1);
                nextCorner[1].x +=wide_Len;
                wide_Len =nextCorner[3].x -nextCorner[2].x;
                wide_Len /=wide_i;
                wide_Len *=(double) (mPtCol -CandidatesXY[2][2] -1);
                nextCorner[2].x -=wide_Len;
            }
            if(i == 1){//y:0-1
                wide_i =CandidatesXY[3][2] -CandidatesXY[1][2] ;
                wide_Len =nextCorner[3].y -nextCorner[0].y;
                wide_Len /=wide_i;
                wide_Len *=(double) CandidatesXY[1][2] ;
                nextCorner[0].y -=wide_Len;
                wide_Len =nextCorner[2].y -nextCorner[1].y;
                wide_Len /=wide_i;
                wide_Len *=(double) CandidatesXY[1][2] ;
                nextCorner[1].y -=wide_Len;
            }
            if(i == 3){//x:3-2
                wide_i =CandidatesXY[3][2] -CandidatesXY[1][2] ;
                wide_Len =nextCorner[3].y -nextCorner[0].y;
                wide_Len /=wide_i;
                wide_Len *=(double) (mPtRow -CandidatesXY[3][2] -1);
                nextCorner[3].y +=wide_Len;
                wide_Len =nextCorner[2].y -nextCorner[1].y;
                wide_Len /=wide_i;
                wide_Len *=(double) (mPtRow -CandidatesXY[3][2] -1);
                nextCorner[2].y -=wide_Len;
            }
        }
/*        if(LogOutput)
            for (int i = 0; i <CandidatesXY.length ; i++) {
                Log.i(TAG, " RectCornerCandidates xy: "
                        +" i:" +i
                        +" nextCorner:" +nextCorner[i]
                );
            }*/
        if(CornerCount >0){
            for (int i = 0; i <nextCorner.length ; i++) {
                nextCorner[i].y =0;
                nextCorner[i].x =0;
            }
        }
        return nextCorner;
    }

    private Point[] ReCreatePoint(Point[] CornerPoint , int mPtRow ,int mPtCol){
        //CornerPoint 0:leftup 1:Rightup 2:RightDown 3:leftdown

        int Pt_i;
/*        Pt_i =mPtRow;
        mPtRow =mPtCol;
        mPtCol= Pt_i;*/
        //長さの予測
        double   PtWideX_Top =(CornerPoint[1].x -CornerPoint[0].x) /(mPtRow -1)//(mPtCol )
                ,PtWideX_Bottom=(CornerPoint[2].x -CornerPoint[3].x) /(mPtRow -1)//(mPtCol )
                ,PtWideY_Top=(CornerPoint[3].y -CornerPoint[0].y) /(mPtCol -1)//(mPtRow )
                ,PtWideY_Bottom =(CornerPoint[2].y -CornerPoint[1].y) /(mPtCol -1)//(mPtRow )
                ,PtWideX_Step =(PtWideX_Bottom -PtWideX_Top) /(mPtCol -1)
                ,PtWideY_Step =(PtWideY_Bottom -PtWideY_Top) /(mPtRow -1)
                ,PtOffsetX_Step =(CornerPoint[3].x -CornerPoint[0].x) /(mPtCol -1)
                ,PtOffsetY_Step =(CornerPoint[1].y -CornerPoint[0].y) /(mPtRow -1)
                ,PtOffsetX =CornerPoint[0].x
                ,PtOffsetY =CornerPoint[0].y ;
/*        if(LogOutput)
            Log.i(TAG, " nextCornerPoint Pt: "
                    +" PtWideX_Top:" +PtWideX_Top
                    +" PtWideX_Bottom:" +PtWideX_Bottom
                    +" PtWideY_Top:" +PtWideY_Top
                    +" PtWideY_Bottom:" +PtWideY_Bottom
                    +" PtWideX_Step:" +PtWideX_Step
                    +" PtWideY_Step:" +PtWideY_Step
                    +" PtOffsetX_Step:" +PtOffsetX_Step
                    +" PtOffsetY_Step:" +PtOffsetY_Step
                    +" PtOffsetX:" +PtOffsetX
                    +" PtOffsetY:" +PtOffsetY
            );*/

        Point [] Candidates_Point =new Point[mPtRow *mPtCol];

        double CandidatesPtX ,CandidatesPtY;
        for (int i = 0; i <mPtRow ; i++) {
            for (int j = 0; j <mPtCol ; j++) {
                Pt_i =i *mPtCol +j;
                CandidatesPtY =(PtWideY_Top +PtWideY_Step *i) *j +PtOffsetY_Step *i +PtOffsetY;
                CandidatesPtX =(PtWideX_Top +PtWideX_Step *j) *i +PtOffsetX_Step *j +PtOffsetX;
                Candidates_Point[Pt_i] =new Point(CandidatesPtX, CandidatesPtY);
            }
        }
/*        if(LogOutput)
            Log.i(TAG, " Corner Pt: " +"\n"
                    +" prevPoint:" +CornerPoint[0]
                    +" nextCandidates_Point:" +Candidates_Point[0] +"\n"
                    +" prevPoint:" +CornerPoint[1]
                    +" nextCandidates_Point:" +Candidates_Point[(mPtRow -1) *mPtCol] +"\n"
                    +" prevPoint:" +CornerPoint[2]
                    +" nextCandidates_Point:" +Candidates_Point[mPtRow *mPtCol -1] +"\n"
                    +" prevPoint:" +CornerPoint[3]
                    +" nextCandidates_Point:" +Candidates_Point[mPtCol -1]
            );*/
        return Candidates_Point;
    }
    private Point[] ReCreateNextPoint(Point[] prevPoint , Point[] nextCornerPoint, int mPtRow ,int mPtCol){
        Point [] prevCorner =new Point[4];
        int Pt_i ;
        for (int i = 0; i <prevCorner.length ; i++) {
            switch (i){
                case 0:
                    Pt_i =0 *0                  ; break;
                case 1:
                    Pt_i =(mPtRow -1) *mPtCol   ; break;
                case 2:
                    Pt_i =mPtRow *mPtCol -1     ; break;
                case 3:
                    Pt_i =mPtCol -1             ; break;
                default:
                    Pt_i =0                     ; break;
            }
            prevCorner[i]=prevPoint[Pt_i];
/*            if(LogOutput)
                Log.i(TAG, " prevCorner next xy: "
                        +" i:" +i
                        +" Pt_i:" +Pt_i
                        +" prevPoint:" +prevPoint[Pt_i]
                );*/
        }
/*        if(LogOutput)
            for (int i = 0; i <prevCorner.length ; i++) {
                Log.i(TAG, " prevCorner xy: "
                        +" i:" +i
                        +" prevCorner:" +prevCorner[i]
                        +" nextCornerPoint:" +nextCornerPoint[i]

                );
            }*/
        //Prev -next time
        Point []PrevNextTime =new Point[4];
        for (int i = 0; i <PrevNextTime.length ; i++) {
            PrevNextTime[i] =new Point(nextCornerPoint[i].x -prevCorner[i].x
                    ,nextCornerPoint[i].y -prevCorner[i].y);
        }
/*        if(LogOutput)
            for (int i = 0; i <prevCorner.length ; i++) {
                Log.i(TAG, " Prev -next time xy: "
                        +" i:" +i
                        +" PrevNextTime:" +PrevNextTime[i]
                );
            }*/

        //長さの予測
        double PtWideX_Top =(PrevNextTime[1].x -PrevNextTime[0].x) /(mPtCol +1)
                ,PtWideX_Bottom=(PrevNextTime[2].x -PrevNextTime[3].x) /(mPtCol +1)
                ,PtWideY_Top=(PrevNextTime[3].y -PrevNextTime[0].y) /(mPtRow +1)
                ,PtWideY_Bottom =(PrevNextTime[2].y -PrevNextTime[1].y) /(mPtRow +1)
                ,PtWideX_Step =(PtWideX_Bottom -PtWideX_Top) /(mPtCol +1)
                ,PtWideY_Step =(PtWideY_Bottom -PtWideY_Top) /(mPtRow +1)
                ,PtOffsetX_Step =((PrevNextTime[3].x -PrevNextTime[0].x) /(mPtCol +1) )
                ,PtOffsetY_Step =((PrevNextTime[1].y -PrevNextTime[0].y) /(mPtRow +1) )
                ,PtOffsetX =PrevNextTime[0].x
                ,PtOffsetY =PrevNextTime[0].y ;
/*        if(LogOutput)
            Log.i(TAG, " nextCornerPoint Pt: "
                    +" PtWideX_Top:" +PtWideX_Top
                    +" PtWideX_Bottom:" +PtWideX_Bottom
                    +" PtWideY_Top:" +PtWideY_Top
                    +" PtWideY_Bottom:" +PtWideY_Bottom
                    +" PtWideX_Step:" +PtWideX_Step
                    +" PtWideY_Step:" +PtWideY_Step
                    +" PtOffsetX_Step:" +PtOffsetX_Step
                    +" PtOffsetY_Step:" +PtOffsetY_Step
                    +" PtOffsetX:" +PtOffsetX
                    +" PtOffsetY:" +PtOffsetY
            );*/
        Point [] nextCandidates_Point =new Point[mPtRow *mPtCol];
        double CandidatesPtX ,CandidatesPtY;
        for (int i = 0; i <mPtRow ; i++) {
            for (int j = 0; j <mPtCol ; j++) {
                Pt_i =i *mPtCol +j;
                CandidatesPtY =prevPoint[Pt_i].y +(PtWideY_Top +PtWideY_Step *i) *j +PtOffsetY_Step *i +PtOffsetY;
                CandidatesPtX =prevPoint[Pt_i].x +(PtWideX_Top +PtWideX_Step *j) *i +PtOffsetX_Step *j +PtOffsetX;
                nextCandidates_Point[Pt_i] =new Point(CandidatesPtX, CandidatesPtY);
            }
        }
/*        if(LogOutput)
            Log.i(TAG, " Corner Pt: " +"\n"
                    +" prevPoint:" +prevPoint[0]
                    +" nextCandidates_Point:" +nextCandidates_Point[0] +"\n"
                    +" prevPoint:" +prevPoint[(mPtRow -1) *mPtCol]
                    +" nextCandidates_Point:" +nextCandidates_Point[(mPtRow -1) *mPtCol] +"\n"
                    +" prevPoint:" +prevPoint[mPtRow *mPtCol -1]
                    +" nextCandidates_Point:" +nextCandidates_Point[mPtRow *mPtCol -1] +"\n"
                    +" prevPoint:" +prevPoint[mPtCol -1]
                    +" nextCandidates_Point:" +nextCandidates_Point[mPtCol -1]
            );*/
        return nextCandidates_Point;
    }

    private Point PointRotate(Point Point ,int Angle ,Mat mMat){
        Point Rotate;
        switch (Angle){
            case 90:
                return Rotate =new Point(Point.y             , mMat.rows()-Point.x);
            case 180:
                return Rotate =new Point(mMat.cols() -Point.x, mMat.rows()-Point.y);
            case 270:
                return Rotate =new Point(mMat.cols() -Point.y,             Point.x);
            default:
                return Rotate =Point.clone();
        }
    }

    private boolean isNearAngle(float SrcAngle ,float TestAngle ,float RangAngle){
        float Theta_i= SrcAngle -TestAngle ;
        if(Theta_i < -180)Theta_i +=360;
        if(Theta_i >= 180)Theta_i -=360;
        if(Theta_i <RangAngle)return true;
        return false;
    }

//    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void SetSoundPool(){
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
            soundPool = new SoundPool(soundList.length, AudioManager.STREAM_ALARM, 0);
        }else{
            audioAttributes = new AudioAttributes.Builder()
                    // USAGE_MEDIA
                    // USAGE_GAME
                    .setUsage(AudioAttributes.USAGE_GAME)
                    // CONTENT_TYPE_MUSIC
                    // CONTENT_TYPE_SPEECH, etc.
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build();

            soundPool = new SoundPool.Builder()
                    .setAudioAttributes(audioAttributes)
                    // ストリーム数に応じて
                    .setMaxStreams(2)
                    .build();
        }


        // Sound をロードしておく
        soundList[0] = soundPool.load(this, R.raw.button50, 1);//動作確認
        soundList[1] = soundPool.load(this, R.raw.button18, 1);//警告速い
        soundList[2] = soundPool.load(this, R.raw.button70, 1);//お知らせ
        soundList[3] = soundPool.load(this, R.raw.button40, 1);//遠方注意
        soundList[4] = soundPool.load(this, R.raw.button44, 1);//近い警告


        // load が終わったか確認する場合
        soundPool.setOnLoadCompleteListener(new SoundPool.OnLoadCompleteListener() {
            @Override
            public void onLoadComplete(SoundPool soundPool, int sampleId, int status) {
//                if(LogOutput)Log.d("debug","sampleId="+sampleId);
//                if(LogOutput)Log.d("debug","status="+status);
            }
        });
    }
    private AudioAttributes audioAttributes;
    private SoundPool soundPool;
    private int [] soundList =new int[5];
    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        if(LogOutput)Log.i(TAG, "called onPrepareOptionsMenu");
        super.onPrepareOptionsMenu(menu);
        menu.clear();
        for (int i = 0; i <ArrayName.length ; i++) {
            if(!ArrayName[i].view)continue;
            String StrTitle = getString( ArrayName[i].Name);
            if (!ArrayName[i].updown)
                menu.add(Menu.NONE, ArrayName[i].Code, Menu.NONE, StrTitle + ":" +String.format("%2.2f", ArrayName[i].value));
            else {
                menu.add(Menu.NONE, ArrayName[i].Code, Menu.NONE, StrTitle +"_" +getString( R.string.up) +":" +String.format("%2.2f", ArrayName[i].value));
                menu.add(Menu.NONE, ArrayName[i].Code +1, Menu.NONE, StrTitle +"_" +getString( R.string.Down) +":" +String.format("%2.2f", ArrayName[i].value));
            }
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if(LogOutput)Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        int i=item.getItemId() -4;
        if(i %2 ==0) {
            i/=2;
            ArrayName[i].value +=ArrayName[i].add;
        }
        else {
            i/=2;
            ArrayName[i].value -=ArrayName[i].add;
        }
        if(ArrayName[i].value <ArrayName[i].min)ArrayName[i].value =ArrayName[i].max;
        if(ArrayName[i].value >ArrayName[i].max)ArrayName[i].value =ArrayName[i].min;
        if(LogOutput)Log.i(TAG, "called onOptionsItemSelected; selected item: i:" + i + " :"+ArrayName[i].value );

        mInterlockingValue(i);
        mSetValue();
        return true;
    }

    private void mSetValue(){
        String Str="";
        for (int i = 0; i <ArrayName.length ; i++) {
            switch (ArrayName[i].Code) {
                case  4://CameraAngle
                    CameraAngle =(int)ArrayName[i].value; break;
                case  6://PtAngle
                    PtAngle =(int)ArrayName[i].value; break;
                case  8://PtSize_x
                    PtSize_x =ArrayName[i].value; break;
                case 10://PtSize_y
                    PtSize_y =ArrayName[i].value; break;
                case 12://PtOffsetPoint_x
                    PtOffsetPoint_x =ArrayName[i].value; break;
                case 14://PtOffsetPoint_y
                    PtOffsetPoint_y =ArrayName[i].value; break;
                case 16://PtAn_x
                    PtAn_x =ArrayName[i].value; break;
                case 18://PtAn_y
                    PtAn_y =ArrayName[i].value; break;
                case 20://AlertLevel
                    AlertLevel =ArrayName[i].value; break;
                case 22://RengeOverTime
                    RangeOverTime =ArrayName[i].value; break;
                case 24://WalkLevel
                    WalkLevel =ArrayName[i].value; break;
                case 26://AlertAreaCount
                    AlertAreaCount =(int)ArrayName[i].value; break;
                case 28://SafetyLevel
                    SafetyLevel =ArrayName[i].value; break;
                case 30://ErrerLevel
                    ErrerLevel =ArrayName[i].value; break;
                case 32://PixelLevel
                    PixelLevel =(int)ArrayName[i].value; break;
                case 34://CheckOff
                    CheckOff =(int)ArrayName[i].value; break;
                case 36://GaussianBlur
                    GaussianBlur =ArrayName[i].value; break;
                case 38://AlertCount
                    AlertCount =ArrayName[i].value; break;
                case 40://BackGroundView
                    BackGroundView =ArrayName[i].value; break;
                case 42://WalkWait
                    WalkWait =ArrayName[i].value; break;
                case 44://Vibration
                    Vibration =ArrayName[i].value; break;
                case 46://Privacypolicy
                    Privacypolicy =ArrayName[i].value; break;
                case 48://Userpolicy
                    Userpolicy =ArrayName[i].value; break;
                case 50://License
                    License =ArrayName[i].value; break;


                default:
                    break;
            }
            PtChange =true;
            Str += " " +getString( ArrayName[i].Name) + ":" +String.format("%2.2f", ArrayName[i].value);
        }
//        if(LogOutput)Log.i(TAG, "called onOptionsItem SetValue: " + Str);
        for (int i = 0; i <ArrayName.length ; i++) {
            SettingValue.mArrayValue[i] =(float) ArrayName[i].value;
        }
        SettingValue.Write();
        if(BackGroundView >0) {
            TextView mTextView =(TextView)findViewById(R.id.textView);
            mTextView.setText(Str);
        }
    }

    private void mInterlockingValue(int Selected){
        switch (ArrayName[Selected].Code){
            case 6:
                switch ((int)ArrayName[Selected].value){
                    case 1:
                        PtAn_x =0.7f;
                        PtAn_y =0.5f;
                        break;
                    case 3:
                        PtAn_x =0.3f;
                        PtAn_y =0.5f;
                        break;
                    case 2:
                        PtAn_x =0.5f;
                        PtAn_y =0.7f;
                        break;
                    default:
                        PtAn_x =0.5f;
                        PtAn_y =0.3f;
                        break;
                }
                ArrayName[6].value =PtAn_x;
                ArrayName[7].value =PtAn_y;
                break;
            case 46:
                mMessage(getString(R.string.PrivacyPolicyTitle)
                        ,getString(R.string.PrivacyPolicy));
                break;
            case 48:
                mMessage(getString(R.string.UserPolicyTitle)
                        ,getString(R.string.UserPolicy));
                break;
            case 50:
                mMessage(getString(R.string.LicenseAgreementTitle)
                        ,getString(R.string.LicenseAgreement));
                break;
        }
    }

    private AlertDialog Dialog;
    public void mMessage(String mTitle,String mMessage){
        AlertDialog.Builder alertDialog =new AlertDialog.Builder(this);
        // ダイアログの設定
        alertDialog.setTitle(mTitle);          //タイトル
        alertDialog.setMessage(mMessage);      //内容
        alertDialog.setPositiveButton("OK",new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int which) {
                //初回表示完了
            }
        });
        alertDialog.create();
        AlertDialog Dialog;
        Dialog = alertDialog.show();
    }

    private static int CameraAngle, PtAngle ;
    private static double PtSize_x, PtSize_y, PtOffsetPoint_x, PtOffsetPoint_y, PtAn_x, PtAn_y;
    private static boolean PtChange =true;
    private static double AlertLevel, WalkLevel, RangeOverTime ,SafetyLevel ,ErrerLevel, AngleNearLevel
            , GaussianBlur, AlertCount, BackGroundView, WalkWait, Vibration
            ,Privacypolicy, Userpolicy, License ;
    private static int AlertAreaCount, CheckOff, PixelLevel;

    public class ButtonNameList {
        int Code ;
        boolean updown,view;
        int Name =0;
        double add, initial, value, min, max;
        private ButtonNameList(int Code_i, int Str ,boolean updown_i , double initial_i, double add_i , double min_i ,double max_i ,boolean view_i) {
            Code   =Code_i;
            updown =updown_i;
            Name   =Str;
            initial =initial_i;
            add    =add_i;
            value  =initial_i;
            min    =min_i;
            max    =max_i;
            view   =view_i;
        }
    }
    public ButtonNameList[] ArrayName = new ButtonNameList[]{
            new ButtonNameList(  4, R.string.mCameraAngle, false, 270, 90, 0f, 270,true),
            new ButtonNameList(  6, R.string.mPtAngle, false, 2, 1, 0f, 3,true),
            new ButtonNameList(  8, R.string.mPtSize_x, true, 1, 0.1f, 0f, 2,true),
            new ButtonNameList( 10, R.string.mPtSize_y, true, 0.7f, 0.1f, 0f, 2,true),
            new ButtonNameList( 12, R.string.mPtOffsetPoint_x, true, 1, 0.05f, 0f, 2,true),
            new ButtonNameList( 14, R.string.mPtOffsetPoint_y, true, 1, 0.05f, 0f, 2,true),
            new ButtonNameList( 16, R.string.mPtAn_x, true, 0.5f, 0.05f, 0f, 1,true),
            new ButtonNameList( 18, R.string.mPtAn_y, true, 0.7f, 0.05f, 0f, 1,true),
            new ButtonNameList( 20, R.string.mAlertLevel, true, 6, 5, 0f, 255,false),
            new ButtonNameList( 22, R.string.mRangeOverTime, true, 5, 5, 0f, 255,false),
            new ButtonNameList( 24, R.string.mWalkLevel, true, 8, 2, 0f, 1000,false),
            new ButtonNameList( 26, R.string.mAlertAreaCount, true, 1, 1, 0f, 1000,true),
            new ButtonNameList( 28, R.string.mSafetyLevel, true, 0.05f, 0.01f, 0f, 1,false),
            new ButtonNameList( 30, R.string.mErrerLevel, true, 0.4f, 0.05f, 0f, 1,false),
            new ButtonNameList( 32, R.string.mPixelLevel, true, 20, 5, 0f, 180,false),
            new ButtonNameList( 34, R.string.mCheckOff, false, 2, 1, 0f, 3,false),
            new ButtonNameList( 36, R.string.mGaussianBlur, true, 31, 2, 1, 100,false),
            new ButtonNameList( 38, R.string.mAlertCount, true, 0f, 1, 0f, 100,false),
            new ButtonNameList( 40, R.string.mBackGroundView, false, 0f, 1, 0f, 1,false),
            new ButtonNameList( 42, R.string.mWalkWait, true, 0f, 0.1f, 0f, 2,true),
            new ButtonNameList( 44, R.string.mVibration, false, 0f, 1, 0f, 1,true),
            new ButtonNameList( 46, R.string.PrivacyPolicyTitle, false, 0f, 1, 0f, 1,true),
            new ButtonNameList( 48, R.string.UserPolicyTitle, false, 0f, 1, 0f, 1,true),
            new ButtonNameList( 50, R.string.LicenseAgreementTitle, false, 0f, 1, 0f, 1,true),

    };
}
