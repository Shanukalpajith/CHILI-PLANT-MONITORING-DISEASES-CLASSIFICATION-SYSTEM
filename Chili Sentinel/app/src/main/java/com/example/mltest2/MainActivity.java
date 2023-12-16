package com.example.mltest2;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.example.mltest2.ml.TfliteModelAnother;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.example.mltest2.ml.TfliteModelAnother;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;




public class MainActivity extends AppCompatActivity {

    TextView result, demoTxt, classified, clickHere;
    ImageView imageView, arrowImage;
    Button picture;

    int imagesize = 224;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main); // Uncomment this line

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View v) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }


/*
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
       // classified = findViewById(R.id.classified);

        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View v) {
                if (checkSelfPermission (Manifest.permission. CAMERA) == PackageManager.PERMISSION_GRANTED){

                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);

                }else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA},100);
                }
            }
        });

    }

 */


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image =(Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image =Bitmap.createScaledBitmap(image, imagesize, imagesize, false);
            classifyImage(image);

        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void classifyImage(Bitmap image) {
        try {
            TfliteModelAnother model = TfliteModelAnother.newInstance(getApplicationContext());

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1,224,224,3}, DataType. FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4* imagesize * imagesize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValue = new int[imagesize * imagesize];
            image.getPixels(intValue, 0, image.getWidth(),  0,  0, image.getWidth(), image.getHeight());

            int pixel = 0;

            for (int i = 0; i < imagesize; i++) {
                for (int j = 0; j < imagesize; j++) {
                    int val = intValue[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            TfliteModelAnother.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeatures0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidence = outputFeatures0.getFloatArray();

            int maxPos = 0;

            float maxConfidence = 0;

            for (int i = 0; i < confidence.length; i++) {

                if (confidence[i] > maxConfidence){
                    maxConfidence = confidence[i];
                    maxPos = i;

                }
            }
            String[] classes = {"Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy"};
            result.setText(classes[maxPos]);
            model.close();

        }catch (IOException e){

        }
    }
}// 12 New
//New12 Final 12/03
// 49, 56,86

