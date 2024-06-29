package ec.edu.ups;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import ec.edu.ups.databinding.ActivityMainBinding;

public class Activity_Main_Part_2 extends AppCompatActivity {
    private android.widget.ImageView imgOriginal, imgModificada, imgHistograma;
    private android.widget.Button btnCamara2, btnClasificar;
    private android.widget.TextView txtRespuesta;
    private Intent intent;
    private ActivityMainBinding binding;
    private Bitmap bitmap, bitmapCalculo;
    static {
        System.loadLibrary("ups");
    }

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main_part2);
        btnCamara2 = findViewById(R.id.btnCamara2);
        btnClasificar = findViewById(R.id.btnClasificar);
        imgOriginal=findViewById(R.id.imgOriginal);
        imgModificada=findViewById(R.id.imgModificada);
        txtRespuesta=findViewById(R.id.txtFiguraDetectada);
        String uriString = getIntent().getStringExtra("imagenUri");
        if (uriString != null) {
            Uri imageUri = Uri.parse(uriString);
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                imgOriginal.setImageBitmap(bitmap);
            } catch (Exception e) {
                //Log.e(TAG, "Error al cargar la imagen desde el URI: " + e.getMessage());
                Toast.makeText(this, "Error al cargar la imagen", Toast.LENGTH_SHORT).show();
            }
        } else {
            //Log.e(TAG, "URI de imagen es null.");
            Toast.makeText(this, "No se recibió ninguna imagen", Toast.LENGTH_SHORT).show();
        }
        btnCamara2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                intent = new Intent(Activity_Main_Part_2.this, Camara_Android_Activity.class);
                startActivity(intent);
            }
        });
        btnClasificar.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            }
        });
    }

}