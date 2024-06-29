#include <jni.h>
#include <string>
//librerias opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>
#include "android/bitmap.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <limits>
#include <opencv2/ml.hpp>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>


using namespace std;
using namespace cv::ml;
using namespace cv;
const int MAX_L = 15;
const double PI = 3.14159265358979323846;
Ptr<SVM> svmModel;
Mat scalerMean, scalerScale;

//momentos zernike
void mb_zernike2D(const cv::Mat &Im, double order, double rad, double *zvalues, long *output_size) {
    int L, N, D;
    N = Im.cols < Im.rows ? Im.cols : Im.rows;
    if (order > 0 && order < MAX_L) L = static_cast<int>(order);
    else L = 15;
    assert(L < MAX_L);

    if (!(rad > 0.0)) rad = N;
    D = static_cast<int>(rad * 2);

    static double H1[MAX_L][MAX_L];
    static double H2[MAX_L][MAX_L];
    static double H3[MAX_L][MAX_L];
    static char init = 1;

    double COST[MAX_L], SINT[MAX_L], R[MAX_L];
    double Rn, Rnm, Rnm2, Rnnm2, Rnmp2, Rnmp4;

    double a, b, x, y, area, r, r2, f, const_t;
    int n, m, i, j;

    double AR[MAX_L][MAX_L], AI[MAX_L][MAX_L];

    double sum = 0;
    int cols = Im.cols;
    int rows = Im.rows;

    double moment10 = 0.0, moment00 = 0.0, moment01 = 0.0;
    double intensity;
    for (i = 0; i < cols; i++) {
        for (j = 0; j < rows; j++) {
            intensity = Im.at<uchar>(j, i);
            sum += intensity;
            moment10 += (i + 1) * intensity;
            moment00 += intensity;
            moment01 += (j + 1) * intensity;
        }
    }
    double m10_m00 = moment10 / moment00;
    double m01_m00 = moment01 / moment00;

    if (init) {
        for (n = 0; n < MAX_L; n++) {
            for (m = 0; m <= n; m++) {
                if (n != m) {
                    H3[n][m] = -(double)(4.0 * (m + 2.0) * (m + 1.0)) / (double)((n + m + 2.0) * (n - m));
                    H2[n][m] = ((double)(H3[n][m] * (n + m + 4.0) * (n - m - 2.0)) / (double)(4.0 * (m + 3.0))) + (m + 2.0);
                    H1[n][m] = ((double)((m + 4.0) * (m + 3.0)) / 2.0) - ((m + 4.0) * H2[n][m]) + ((double)(H3[n][m] * (n + m + 6.0) * (n - m - 4.0)) / 8.0);
                }
            }
        }
        init = 0;
    }

    for (n = 0; n <= L; n++) {
        for (m = 0; m <= n; m++) {
            AR[n][m] = AI[n][m] = 0.0;
        }
    }

    area = PI * rad * rad;
    for (i = 0; i < cols; i++) {
        x = (i + 1 - m10_m00) / rad;
        for (j = 0; j < rows; j++) {
            y = (j + 1 - m01_m00) / rad;
            r2 = x * x + y * y;
            r = sqrt(r2);
            if (r < DBL_EPSILON || r > 1.0) continue;

            R[0] = 1;
            for (n = 1; n <= L; n++) R[n] = r * R[n - 1];

            a = COST[0] = x / r;
            b = SINT[0] = y / r;
            for (m = 1; m <= L; m++) {
                COST[m] = a * COST[m - 1] - b * SINT[m - 1];
                SINT[m] = a * SINT[m - 1] + b * COST[m - 1];
            }

            f = Im.at<uchar>(j, i) / sum;

            Rnmp2 = Rnm2 = 0;
            for (n = 0; n <= L; n++) {
                const_t = (n + 1) * f / PI;
                Rn = R[n];
                if (n >= 2) Rnm2 = R[n - 2];
                for (m = n; m >= 0; m -= 2) {
                    if (m == n) {
                        Rnm = Rn;
                        Rnmp4 = Rn;
                    } else if (m == n - 2) {
                        Rnnm2 = n * Rn - (n - 1) * Rnm2;
                        Rnm = Rnnm2;
                        Rnmp2 = Rnnm2;
                    } else {
                        Rnm = H1[n][m] * Rnmp4 + (H2[n][m] + (H3[n][m] / r2)) * Rnmp2;
                        Rnmp4 = Rnmp2;
                        Rnmp2 = Rnm;
                    }
                    AR[n][m] += const_t * Rnm * COST[m];
                    AI[n][m] -= const_t * Rnm * SINT[m];
                }
            }
        }
    }

    int numZ = 0;
    for (n = 0; n <= L; n++) {
        for (m = 0; m <= n; m++) {
            if ((n - m) % 2 == 0) {
                AR[n][m] *= AR[n][m];
                AI[n][m] *= AI[n][m];
                zvalues[numZ] = fabs(sqrt(AR[n][m] + AI[n][m]));
                numZ++;
            }
        }
    }
    *output_size = numZ;
}

double distanciaEuclidea(double* zvalues, double* referencia, int size) {
    double suma = 0.0;
    for (int i = 0; i < size; ++i) {
        suma += pow(zvalues[i] - referencia[i], 2);
    }
    return sqrt(suma);
}

string clasificarFiguraZernike(double* zvalues, int size, double* referenciaCirculo, double* referenciaCuadrado, double* referenciaTriangulo) {
    long outputSizeCirculo = 10;
    long outputSizeCuadrado = 10;
    long outputSizeTriangulo = 10;
    double distanciaCirculo = distanciaEuclidea(zvalues, referenciaCirculo, size);
    double distanciaCuadrado = distanciaEuclidea(zvalues, referenciaCuadrado, size);
    double distanciaTriangulo = distanciaEuclidea(zvalues, referenciaTriangulo, size);

    if (distanciaCirculo < distanciaCuadrado && distanciaCirculo < distanciaTriangulo) {
        return "Circulo";
    } else if (distanciaCuadrado < distanciaCirculo && distanciaCuadrado < distanciaTriangulo) {
        return "Cuadrado";
    } else if (distanciaTriangulo < distanciaCirculo && distanciaTriangulo < distanciaCuadrado) {
        return "Triangulo";
    } else {
        return "Es otra figura";
    }
}
//momentos hu
double distanciaEuclidea(double momentosHu[7], double referenciaHu[7]){
    double suma = 0;
    for(int i = 0; i < 7; i++){
        suma += pow(momentosHu[i] - referenciaHu[i], 2);
    }
    return sqrt(suma);
}
string clasificarFigura(double momentosHu[7]) {
    double huReferenciaCirculo[7] = {0.166, 0.0002, 0.0002, 0.0000001, 0.0000001, 0.0000001, 0.0000001};
    double huReferenciaCuadrado[7] = {1.88995251e-01, 4.71506782e-03, 1.55958718e-04, 5.48292604e-05, 7.21290167e-08, 2.56451470e-06, -9.99378394e-10};
    double huReferenciaTriangulo[7] = {0.219, 0.001, 0.0005, 0.0000001, 0.0000001, 0.0000001, 0.0000001};
    double distanciaCirculo = distanciaEuclidea(momentosHu, huReferenciaCirculo);
    double distanciaCuadrado = distanciaEuclidea(momentosHu, huReferenciaCuadrado);
    double distanciaTriangulo = distanciaEuclidea(momentosHu, huReferenciaTriangulo);

    if (distanciaCirculo < distanciaCuadrado && distanciaCirculo < distanciaTriangulo) {
        return "Circulo";
    } else if (distanciaCuadrado < distanciaCirculo && distanciaCuadrado < distanciaTriangulo) {
        return "Cuadrado";
    } else if(distanciaTriangulo < distanciaCirculo && distanciaTriangulo < distanciaCuadrado){
        return "Triangulo";
    } else{
        return "Es otra  figura";
    }
}

void bitmapToMat(JNIEnv * env, jobject bitmap, cv::Mat &dst, jboolean needUnPremultiplyAlpha){
    AndroidBitmapInfo info;
    void* pixels = 0;
    try {
        CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
        CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                   info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
        CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
        CV_Assert( pixels );
        dst.create(info.height, info.width, CV_8UC4);
        if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
        {
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(needUnPremultiplyAlpha) cvtColor(tmp, dst, cv::COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
// info.format == ANDROID_BITMAP_FORMAT_RGB_565
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, cv::COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch(const cv::Exception& e) {
        AndroidBitmap_unlockPixels(env, bitmap);
//jclass je = env->FindClass("org/opencv/core/CvException");
        jclass je = env->FindClass("java/lang/Exception");
//if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}
void matToBitmap(JNIEnv * env, cv::Mat src, jobject bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void* pixels = 0;
    try {
        CV_Assert( AndroidBitmap_getInfo(env, bitmap, &info) >= 0 );
        CV_Assert( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                   info.format == ANDROID_BITMAP_FORMAT_RGB_565 );
        CV_Assert( src.dims == 2 && info.height == (uint32_t)src.rows && info.width == (uint32_t)src.cols );
        CV_Assert( src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4 );
        CV_Assert( AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0 );
        CV_Assert( pixels );
        if( info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 )
        {
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if(src.type() == CV_8UC1)
            {
                cvtColor(src, tmp, cv::COLOR_GRAY2RGBA);
            } else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, cv::COLOR_RGB2RGBA);
            } else if(src.type() == CV_8UC4){
                if(needPremultiplyAlpha) cvtColor(src, tmp, cv::COLOR_RGBA2mRGBA);
                else src.copyTo(tmp);
            }
        } else {
// info.format == ANDROID_BITMAP_FORMAT_RGB_565
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if(src.type() == CV_8UC1)
            {
                cvtColor(src, tmp, cv::COLOR_GRAY2BGR565);
            } else if(src.type() == CV_8UC3){
                cvtColor(src, tmp, cv::COLOR_RGB2BGR565);
            } else if(src.type() == CV_8UC4){
                cvtColor(src, tmp, cv::COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch(const cv::Exception& e) {
        AndroidBitmap_unlockPixels(env, bitmap);
//jclass je = env->FindClass("org/opencv/core/CvException");
        jclass je = env->FindClass("java/lang/Exception");
//if(!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}
//codigo para presentar los diez numeros de fibonachi
extern "C" JNIEXPORT jstring JNICALL
Java_ups_edu_ec_aplicacionnativa_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    int a = 0;
    int b = 1;
    int c = 0;
    stringstream ss;
    ss << a << "," << b << ",";
    for (int i=0;i<10;i++){
        c = a + b;
        a = b;
        b = c;
        ss << c << ",";
    }
    return env->NewStringUTF(ss.str().c_str());
}
//codigo para procesamiento de la imagen tomada
extern "C" JNIEXPORT void JNICALL
Java_ec_edu_ups_MainActivity_procesarImagen(
        JNIEnv* env,
        jobject /*this*/,
        jobject bitmapIn,
        jobject bitmapOut){
cv::Mat src;
cv::Mat tmp;
cv::Mat imagenGray;
cv::Mat imagenInvertida;
cv::Mat imagenThreshold;
cv::Mat imagenProcesada;
bitmapToMat(env, bitmapIn, src, false);
cv::medianBlur(src, tmp,1);
cv::cvtColor(tmp,imagenGray,cv::COLOR_BGR2GRAY);
cv::threshold(imagenGray, imagenThreshold,100,255,0);
cv::bitwise_not(imagenThreshold, imagenInvertida);
cv::floodFill(imagenInvertida, cv::Point(src.cols/2,src.rows/2),cv::Scalar(255,0,255));
imagenInvertida.copyTo(imagenInvertida);
cv::convertScaleAbs(imagenInvertida, imagenInvertida);
matToBitmap(env, imagenInvertida, bitmapOut, false);
}
//codigo para calcular los momentos de hu
extern "C" JNIEXPORT jstring JNICALL
Java_ec_edu_ups_MainActivity_momentosHu(
        JNIEnv* env,
jobject /*this*/,
jobject bitmapIn
        ){
    cv::Mat src;
    cv::Mat src_gray;
    stringstream ss;
    // Convertir bitmap a Mat
    bitmapToMat(env, bitmapIn, src, false);
    cv::cvtColor(src, src_gray, cv::COLOR_RGBA2GRAY); // o cv::COLOR_BGR2GRAY dependiendo del formato de color del bitmap
// Calcular momentos de Hu
    cv::Moments momentosImagen = moments(src_gray, true);
    double momentosHuImagen[7];
    cv::HuMoments(momentosImagen, momentosHuImagen);
    // Clasificar la figura basada en los momentos de Hu
    std::string figura = clasificarFigura(momentosHuImagen);
    // Preparar el resultado para enviar de vuelta a Java
    ss <<"La figura detectada es: "<< figura <<endl;
    ss << "Momentos de Hu: ";
    for (int i = 0; i < 7; i++) {
        ss << momentosHuImagen[i] << ", "<<endl;
    }
    // Devolver la cadena como jstring
    return env->NewStringUTF(ss.str().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_ec_edu_ups_MainActivity_momentosZernike(
        JNIEnv* env,
        jobject /*this*/,
        jobject bitmapIn
){
    cv::Mat src;
    cv::Mat imagenObjeto;
    stringstream ss;
    // Valores predefinidos de los momentos de Zernike para cada figura
    double zernikeValuesCirculo[] = {0.00150965, 0.0617555, 0.00409167, 0.0294977, 0.0452987, 0.0212328, 0.0214953, 0.00982531, 0.0188014, 0.0284563 ,0.0255214,
                                     0.00394764, 0.0308236, 0.0114317, 0.0171248, 0.00565382, 0.0688947, 0.0121268, 0.0171143, 0.0109251, 0.0168657, 0.00595771, 0.00228536, 0.0169846, 0.0044488,
                                     0.0337955, 0.0176261, 0.015828, 0.00654511, 0.0150399, 0.0122085, 0.0178828, 0.0156046, 0.010641, 0.00534109, 0.0044142, 0.0281548, 0.0286395 ,0.00463315,
                                     0.0194586, 0.00624491, 0.0122154, 0.0259019, 0.0161305, 0.00674624, 0.010169, 0.0114711, 0.00447335, 0.00255231, 0.0405218, 0.0109971, 0.0103681, 0.0134532,
                                     0.0135346, 0.00421728, 0.00872937, 0.0176717, 0.0112034, 0.0126626, 0.0174722, 0.00737336, 0.00358257, 0.00987377, 0.00041515};
    double zernikeValuesCuadrado[] = {0.00226296, 0.0988681, 0.00141188, 0.0134589, 0.0172446, 0.0534104, 0.00973962, 0.0115116, 0.0160072, 0.0989596, 0.0226333,
                                      0.0892062, 0.00689847, 0.0121889, 0.00373962, 0.0168099, 0.0320035, 0.00956434, 0.00651139, 0.0235891, 0.00694929, 0.00208144, 0.0145198, 0.00585856, 0.0137973,
                                      0.0553476, 0.0250828, 0.0419275, 0.0222515, 0.0337334, 0.00796064, 0.00515843, 0.00409757, 0.0151865, 0.0036052, 0.0172135, 0.0217344, 0.0263086, 0.00857994,
                                      0.0318183, 0.0228759, 0.021801, 0.0021728, 0.00685666, 0.0138352, 0.00602846, 0.0170841, 0.0034195, 0.0258922, 0.0205794, 0.0351094, 0.0145093, 0.0238506,
                                      0.00351993, 0.0221468, 0.0170573, 0.00543334, 0.00367948, 0.00705744, 0.010648, 0.00409955, 0.0108467, 0.0029405, 0.00934178};
    double zernikeValuesTriangulo[] = {0.00912938, 0.056422, 0.0166479, 0.139476, 0.0342623, 0.03022, 0.0216766, 0.109231, 0.0858091, 0.0233744, 0.0454884,
                                       0.0745128, 0.0275008, 0.0271693, 0.061416, 0.0297408, 0.0451553, 0.0114846, 0.0459297, 0.0913794, 0.00532849, 0.0810979, 0.0146289, 0.0392461, 0.0260714,
                                       0.0170202, 0.0262568, 0.0151969, 0.0575276, 0.0221676, 0.0281622, 0.0305654, 0.0548421, 0.0129719, 0.00700524, 0.079748, 0.0247289, 0.0186442, 0.0407473,
                                       0.0180693, 0.02103, 0.0128434, 0.0230988, 0.0291421, 0.0235601, 0.014187, 0.010053, 0.0395777, 0.0212003, 0.0363995, 0.0136012, 0.00927283, 0.0610495,
                                       0.0168351, 0.0237698, 0.0564536, 0.00812797, 0.0461964, 0.0224877, 0.0171761, 0.0121729, 0.0192847, 0.00922147, 0.0274908 };
    // Convertir bitmap a Mat
    bitmapToMat(env, bitmapIn, src, false);

    cv::cvtColor(src, imagenObjeto, cv::COLOR_RGBA2GRAY); // o cv::COLOR_BGR2GRAY dependiendo del formato de color del bitmap

// Calcular momentos de Zernike
    double zernikeValuesObjeto[100];
    long outputSizeObjeto;
    mb_zernike2D(imagenObjeto, 14, imagenObjeto.cols / 2.0, zernikeValuesObjeto, &outputSizeObjeto);
    string figura = clasificarFiguraZernike(zernikeValuesObjeto, outputSizeObjeto, zernikeValuesCirculo, zernikeValuesCuadrado, zernikeValuesTriangulo);
    ss <<"La figura detectada es: "<< figura <<endl;
    ss << "Momentos de Zernike: ";
    for (int i = 0; i < outputSizeObjeto; i++) {
        ss << zernikeValuesObjeto[i] << ", "<<endl;
    }

    // Devolver la cadena como jstring
    return env->NewStringUTF(ss.str().c_str());
}

//parte 2
void loadScaler(const string& filepath, Mat& mean, Mat& scale) {
    FileStorage fs(filepath, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: could not open the scaler file!" << endl;
        return;
    }
    fs["mean"] >> mean;
    fs["scale"] >> scale;
    fs.release();

    // Convertir mean y scale a una fila
    mean = mean.reshape(1, 1);
    scale = scale.reshape(1, 1);

    // Convertir mean y scale a float
    mean.convertTo(mean, CV_32F);
    scale.convertTo(scale, CV_32F);
}

// Función para escalar características utilizando el escalador cargado
Mat scaleFeatures(const Mat& features, const Mat& mean, const Mat& scale) {
    Mat scaledFeatures;
    subtract(features, mean, scaledFeatures);
    divide(scaledFeatures, scale, scaledFeatures);
    return scaledFeatures;
}

// Función para extraer características LBP de una imagen Mat
Mat extractLBPFeatures(const Mat& src) {
    Mat lbpImage = Mat::zeros(src.size(), CV_8UC1);
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            uchar center = src.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (src.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (src.at<uchar>(i - 1, j) > center) << 6;
            code |= (src.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (src.at<uchar>(i, j + 1) > center) << 4;
            code |= (src.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (src.at<uchar>(i + 1, j) > center) << 2;
            code |= (src.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (src.at<uchar>(i, j - 1) > center) << 0;
            lbpImage.at<uchar>(i - 1, j - 1) = code;
        }
    }

    Mat hist = Mat::zeros(1, 10, CV_32SC1); // Ajuste el histograma a 10 características
    for (int i = 0; i < lbpImage.rows; i++) {
        for (int j = 0; j < lbpImage.cols; j++) {
            hist.at<int>(0, lbpImage.at<uchar>(i, j))++;
        }
    }

    hist.convertTo(hist, CV_32F);
    hist = hist / sum(hist)[0];
    return hist;
}

// Función para predecir la clase de una imagen utilizando el modelo SVM
string predictImage(const Mat& image, const Mat& mean, const Mat& scale) {
    Mat gray;
    cvtColor(image, gray, COLOR_RGBA2GRAY);  // Convertir a escala de grises si es necesario

    Mat features = extractLBPFeatures(gray);
    features = features.reshape(1, 1); // Asegúrate de que las características sean una fila
    features = scaleFeatures(features, mean, scale);

    float response = svmModel->predict(features);

    return response == 1.0 ? "Metal" : "Madera";
}

// Función para cargar el modelo SVM desde un archivo XML
bool loadSVMModel(const string& modelPath) {
    svmModel = SVM::load(modelPath);
    if (svmModel.empty()) {
        cerr << "Error: could not load the SVM model from " << modelPath << endl;
        return false;
    }
    return true;
}

// Función para cargar el escalador y el modelo SVM
extern "C" JNIEXPORT void JNICALL
Java_ec_edu_ups_MainActivity_loadModel(JNIEnv* env, jobject /*this*/, jstring modelPath, jstring scalerPath) {
    const char *model_path = env->GetStringUTFChars(modelPath, nullptr);
    const char *scaler_path = env->GetStringUTFChars(scalerPath, nullptr);

    bool modelLoaded = loadSVMModel(model_path);
    if (modelLoaded) {
        loadScaler(scaler_path, scalerMean, scalerScale);
    }

    env->ReleaseStringUTFChars(modelPath, model_path);
    env->ReleaseStringUTFChars(scalerPath, scaler_path);
}

// Función para clasificar una imagen Mat utilizando SVM y LBP
extern "C" JNIEXPORT jstring JNICALL
Java_ec_edu_ups_MainActivity_classifyImage(JNIEnv* env, jobject /*this*/, jobject bitmap) {
    cv::Mat img;
    bitmapToMat(env, bitmap, img, false);  // Convertir el bitmap a Mat

    string result = predictImage(img, scalerMean, scalerScale);

    return env->NewStringUTF(result.c_str());
}
