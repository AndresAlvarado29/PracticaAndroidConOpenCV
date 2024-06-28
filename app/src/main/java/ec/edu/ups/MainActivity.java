package ec.edu.ups;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import ec.edu.ups.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {
    private android.widget.Button btnCamara, btnHu,btnPart2,btnZernike,btnProcesar;
    private android.widget.TextView txtMomentos,txtFiguraDetectada;
    private android.widget.ImageView imgOriginal,imgModificado;
    private Intent intent;
    private ActivityMainBinding binding;
    private Bitmap bitmap, bitmapCalculo;
    // Used to load the 'ups' library on application startup.
    static {
        System.loadLibrary("ups");
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
    btnCamara = findViewById(R.id.btnCamara);
    imgOriginal=findViewById(R.id.imageOriginal);
    imgModificado=findViewById(R.id.imageModificada);
    txtFiguraDetectada=findViewById(R.id.txtFiguraDetectada);
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
    btnCamara.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
           intent = new Intent(MainActivity.this, Camara_Android_Activity.class);
           startActivity(intent);
        }
    });

    btnProcesar = findViewById(R.id.btnProcesar);
    btnProcesar.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            android.graphics.Bitmap bIn = bitmap;
            android.graphics.Bitmap bOut = bIn.copy(bIn.getConfig(), true);
            procesarImagen(bIn, bOut);
            imgModificado.setImageBitmap(bOut);
            bitmapCalculo=bOut;
        }
    });
    btnHu = findViewById(R.id.btnHu);
    btnHu.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            android.graphics.Bitmap bIn = bitmapCalculo;
            txtFiguraDetectada.setText(momentosHu(bIn));
        }
    });
    btnZernike = findViewById(R.id.btnZernike);
    btnZernike.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            android.graphics.Bitmap bIn = bitmapCalculo;
            txtFiguraDetectada.setText(momentosZernike(bIn));
        }
    });
    btnPart2 = findViewById(R.id.btnPart2);
    btnPart2.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            intent = new Intent(MainActivity.this, Activity_Main_Part_2.class);
            startActivity(intent);
        }
    });


    }

    /**
     * A native method that is implemented by the 'ups' native library,
     * which is packaged with this application.
     */
   // public native String stringFromJNI();
    public native void procesarImagen(android.graphics.Bitmap in, android.graphics.Bitmap out);
    public native String momentosHu(android.graphics.Bitmap in);
    public native String momentosZernike(android.graphics.Bitmap in);
}