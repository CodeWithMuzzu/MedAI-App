package com.example.myapplication;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.myapplication.ml.BrainModel;
import com.example.myapplication.ml.EyeModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;


public class NeurologyFragment extends Fragment {

        Button selectBtn , predictBtn, captureBtn;
        TextView result;
        ImageView imageView;
        Bitmap bitmap;

        @Override
        public void onCreate(Bundle savedInstanceState){
            super.onCreate(savedInstanceState);
        }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View v =  inflater.inflate(R.layout.fragment_neurology, container, false);

        getPermission();

        imageView =  v.findViewById(R.id.nImageView);
        selectBtn = v.findViewById(R.id.nUploadImage);
        captureBtn = v.findViewById(R.id.nCapture);
        predictBtn = v.findViewById(R.id.nPredict);
        result = v.findViewById(R.id.nOutput);




        // Inflate the layout for this fragment


        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });

        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);
            }
        });

        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

// Convert the resized bitmap to a byte buffer
                int numBytes = resizedBitmap.getByteCount();
                ByteBuffer byteBuffer = ByteBuffer.allocate(numBytes);
                resizedBitmap.copyPixelsToBuffer(byteBuffer);
                byteBuffer.rewind();

                try {
                    // Load the TFLite model and process the input
                    BrainModel model = BrainModel.newInstance(getActivity());
                    // Create a TensorBuffer with the desired shape
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

// Get the current ByteBuffer data from the TensorBuffer
                    byteBuffer = inputFeature0.getBuffer();

// Create a new ByteBuffer with the desired capacity
                    int newCapacity = 1 * 224 * 224 * 3 * 4; // 1 x 256 x 192 x 3 x 4 bytes per float
                    ByteBuffer newByteBuffer = ByteBuffer.allocate(newCapacity);

// Copy the contents of the original buffer into the new buffer
                    byteBuffer.rewind();
                    newByteBuffer.put(byteBuffer);

                    Log.d("shape","inputFeature0 shape: " + Arrays.toString(inputFeature0.getShape()));
                    Log.d("shape","byteBuffer capacity: " + byteBuffer.capacity());


// Set the new buffer as the data for the input feature buffer
                    inputFeature0.loadBuffer(newByteBuffer);


                    BrainModel.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Get the output probability and display it in the text view
                    float[] probability = outputFeature0.getFloatArray();

                    int index = 0;
                    float max = 0;
                    for(int i= 0;i<probability.length;i++){
                        if(probability[i] > max){
                            max = probability[i];
                            index = i;
                        }
                    }

                    String[] classes = {"glioma tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"};
                    result.setText(classes[index]);
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });

        return v;
    }

    void getPermission(){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if(ContextCompat.checkSelfPermission(getActivity(), android.Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED){
                ActivityCompat.requestPermissions(getActivity(), new String[]{android.Manifest.permission.CAMERA}, 11);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode == 11){
            if(grantResults.length>0){
                if(grantResults[0] == PackageManager.PERMISSION_GRANTED){
                    this.getPermission();
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode==10){
            if(data!=null){
                Uri uri  = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(getActivity().getContentResolver(), uri);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

            }
        }else if(requestCode == 12){
            bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}