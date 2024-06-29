package ec.edu.ups;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;


public class Camara_Android_Activity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String  TAG = "OCVSample::Activity";
    private CameraBridgeViewBase camaraActivity;
    private String nombre="Foto_";
    private Mat frame;
    private android.widget.Button btnTomar,btnTomar2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully");
        } else {
            Log.e(TAG, "OpenCV initialization failed!");
            (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show();
            return;
        }
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camara_android);
        camaraActivity = (CameraBridgeViewBase) findViewById(R.id.camaraVista);
        camaraActivity.setVisibility(SurfaceView.VISIBLE);
        camaraActivity.setCvCameraViewListener(this);
        //boton
        btnTomar=findViewById(R.id.btnTomarFoto);
        btnTomar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                        != PackageManager.PERMISSION_GRANTED || checkSelfPermission(android.Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                    String[] permissions = {android.Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA};
                    requestPermissions(permissions, 1);
                }else{
                    Log.d(TAG, "Permisos ya concedidos. Capturando imagen.");
                    capturarImagen(frame);
                    Uri imagenUri = almacenar(frame);
                    Intent intent = new Intent(Camara_Android_Activity.this, MainActivity.class);
                    intent.putExtra("imagenUri", imagenUri.toString());
                    startActivity(intent);
                }
            }
        });
        btnTomar2=findViewById(R.id.btnTomarFotoPart2);
        btnTomar2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                        != PackageManager.PERMISSION_GRANTED || checkSelfPermission(android.Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
                    String[] permissions = {android.Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA};
                    requestPermissions(permissions, 1);
                }else{
                    Log.d(TAG, "Permisos ya concedidos. Capturando imagen.");
                    capturarImagen(frame);
                    Uri imagenUri = almacenar(frame);
                    Intent intent = new Intent(Camara_Android_Activity.this, Activity_Main_Part_2.class);
                    intent.putExtra("imagenUri", imagenUri.toString());
                    startActivity(intent);
                }
            }
        });
    }

    public Bitmap capturarImagen(Mat frame){
        if (frame == null) {
            Log.e(TAG, "Frame es null, no se puede capturar imagen.");
            return null;
        }
        //Core.rotate(frame,frame, Core.ROTATE_90_CLOCKWISE);
        Bitmap bitmap= Bitmap.createBitmap(frame.cols(),frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame,bitmap);
        return bitmap;
    }
    public Uri almacenar(Mat frame){
        if (frame == null) {
            Log.e(TAG, "Frame es null, no se puede capturar imagen.");
            return null;
        }
        Core.rotate(frame,frame, Core.ROTATE_90_CLOCKWISE);
        Bitmap bitmap= Bitmap.createBitmap(frame.cols(),frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame,bitmap);
        nombre=nombre+new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()) + ".png";
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, nombre);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES);
        Uri uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        try {
            if (uri != null) {
                OutputStream outputStream = getContentResolver().openOutputStream(uri);
                if (outputStream != null) {
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
                    outputStream.close();
                    return uri;
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Error al guardar la imagen: " + e.getMessage());
        }

        return null;
    }
    @Override
    public void onPause()
    {
        super.onPause();
        if (camaraActivity != null)
            camaraActivity.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (camaraActivity != null)
            camaraActivity.enableView();
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(camaraActivity);
    }
    @Override
    public void onDestroy() {
        super.onDestroy();
        if (camaraActivity != null)
            camaraActivity.disableView();
    }
    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        frame = inputFrame.rgba();
        return frame;
    }
    }