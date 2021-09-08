// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

import static android.provider.MediaStore.Files.FileColumns.MEDIA_TYPE_IMAGE;

public class MainActivity extends AppCompatActivity implements Runnable {

    static {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(new SystemDelegate());
        }
        NativeLoader.loadLibrary("pytorch_jni");
        NativeLoader.loadLibrary("torchvision_ops");
    }

    private Camera mCamera;
    private CameraPreview mPreview;

    private TextView mCountView;
    private TextView mTotalView;

    private Button mButtonDetect;
    private Button mButtonAdd;
    private Button mButtonDiscard;
    private Button mButtonReset;
    private ProgressBar mProgressBar;
    private ResultView mResultView;

    private Bitmap mBitmap = null;
    private Module mModule = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    public static final String TAG = "Camera";
    public static final int MEDIA_TYPE_IMAGE = 1;
    public static final int MEDIA_TYPE_VIDEO = 2;

    public int totalCount = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open("5.JPG"));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mCountView = findViewById(R.id.countView);
        mTotalView = findViewById(R.id.totalView);

        mButtonDetect = findViewById(R.id.detectButton);
        mButtonAdd = findViewById(R.id.addButton);
        mButtonDiscard = findViewById(R.id.discardButton);
        mButtonReset = findViewById(R.id.resetButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);

        //Camera Setup
        mCamera = getCameraInstance();
        mPreview = new CameraPreview(this, mCamera);
        FrameLayout preview = (FrameLayout) findViewById(R.id.camera_preview);
        preview.addView(mPreview);

        //Results
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);


        //OnCLickListeners
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mCamera.takePicture(null, null, mPicture);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.INPUT_WIDTH;
                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.INPUT_HEIGHT;

                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mPreview.getWidth() / mBitmap.getWidth() : (float)mPreview.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mPreview.getHeight() / mBitmap.getHeight() : (float)mPreview.getWidth() / mBitmap.getWidth());

                mStartX = (mPreview.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mPreview.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        mButtonAdd.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                totalCount += Integer.parseInt((String) mCountView.getText());
                mCountView.setText("0");
                mTotalView.setText(String.valueOf(totalCount));

                mCamera.stopPreview();
                mCamera.startPreview();

                mButtonDetect.setEnabled(true);
                mButtonDetect.setText(getString(R.string.detect));

                mResultView.setVisibility(View.INVISIBLE);
            }
        });

        mButtonDiscard.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mCountView.setText("0");

                mCamera.stopPreview();
                mCamera.startPreview();

                mButtonDetect.setEnabled(true);
                mButtonDetect.setText(getString(R.string.detect));

                mResultView.setVisibility(View.INVISIBLE);
            }
        });

        mButtonReset.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                totalCount = 0;
                mTotalView.setText(String.valueOf(totalCount));
            }
        });

        try {
            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "d2go.pt");

            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
    }


    @Override
    public void run() {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.INPUT_WIDTH, PrePostProcessor.INPUT_HEIGHT, true);

        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * resizedBitmap.getWidth() * resizedBitmap.getHeight());
        TensorImageUtils.bitmapToFloatBuffer(resizedBitmap, 0,0,resizedBitmap.getWidth(),resizedBitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
        final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[] {3, resizedBitmap.getHeight(), resizedBitmap.getWidth()});

        final long startTime = SystemClock.elapsedRealtime();
        IValue[] outputTuple = mModule.forward(IValue.listFrom(inputTensor)).toTuple();
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("D2Go",  "inference time (ms): " + inferenceTime);

        final Map<String, IValue> map = outputTuple[1].toList()[0].toDictStringKey();
        float[] boxesData = new float[]{};
        float[] scoresData = new float[]{};
        long[] labelsData = new long[]{};
        if (map.containsKey("boxes")) {
            final Tensor boxesTensor = map.get("boxes").toTensor();
            final Tensor scoresTensor = map.get("scores").toTensor();
            final Tensor labelsTensor = map.get("labels").toTensor();
            boxesData = boxesTensor.getDataAsFloatArray();
            scoresData = scoresTensor.getDataAsFloatArray();
            labelsData = labelsTensor.getDataAsLongArray();

            final int n = scoresData.length;
            float[] outputs = new float[n * PrePostProcessor.OUTPUT_COLUMN];
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (scoresData[i] < 0.5)
                    continue;

                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 0] = boxesData[4 * i + 0];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 1] = boxesData[4 * i + 1];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 2] = boxesData[4 * i + 2];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 3] = boxesData[4 * i + 3];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 4] = scoresData[i];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 5] = labelsData[i] - 1;
                count++;
            }

            final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
            final String stringCount = String.valueOf(count);

            runOnUiThread(() -> {
//                mButtonDetect.setEnabled(true);
//                mButtonDetect.setText(getString(R.string.detect));
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);

                //Count
                mCountView.setText(stringCount);

//                mPreview.setVisibility(View.INVISIBLE);

                //Bounding Boxes
                mResultView.setResults(results);
                mResultView.bringToFront();
                mResultView.invalidate();
                mResultView.setVisibility(View.VISIBLE);
            });
        }
    }


    //Camera Methods
    public static Camera getCameraInstance(){
        Camera c = null;
        try {
            c = Camera.open(); // attempt to get a Camera instance
        }
        catch (Exception e){
            // Camera is not available (in use or does not exist)
        }
        return c; // returns null if camera is unavailable
    }


    private Camera.PictureCallback mPicture = new Camera.PictureCallback() {
        @Override
        public void onPictureTaken(byte[] data, Camera camera) {
            Log.d("Camera", "onPictureTaken called");
            File pictureFile = getOutputMediaFile(MEDIA_TYPE_IMAGE);
            if (pictureFile == null){
                Log.d("Camera", "Error creating media file, check storage permissions");
                return;
            }

            try {
                FileOutputStream fos = new FileOutputStream(pictureFile);
                fos.write(data);
                fos.close();
                Log.d("Camera", "File Written");
            } catch (FileNotFoundException e) {
                Log.d("Camera", "File not found: " + e.getMessage());
            } catch (IOException e) {
                Log.d("Camera", "Error accessing file: " + e.getMessage());
            }

            try {
                mBitmap = BitmapFactory.decodeStream(new FileInputStream(pictureFile));
                Log.d("Camera", "Loaded bitmap");
                int w = mBitmap.getWidth();
                int h = mBitmap.getHeight();
                Log.d("Camera", String.valueOf(w) + "  " + String.valueOf(h));
            } catch (IOException e) {
                Log.e("Camera", "Error reading assets", e);
                finish();
            }

//            mCamera.stopPreview();
//            mCamera.startPreview();
        }
    };


    /** Create a file Uri for saving an image or video */
    private static Uri getOutputMediaFileUri(int type){
        return Uri.fromFile(getOutputMediaFile(type));
    }


    /** Create a File for saving an image or video */
    private static File getOutputMediaFile(int type){
        // To be safe, you should check that the SDCard is mounted
        // using Environment.getExternalStorageState() before doing this.

        File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), "MyCameraApp");
        // This location works best if you want the created images to be shared
        // between applications and persist after your app has been uninstalled.

        // Create the storage directory if it does not exist
        if (! mediaStorageDir.exists()){
            if (! mediaStorageDir.mkdirs()){
                Log.d("MyCameraApp", "failed to create directory");
                return null;
            }
        }

        // Create a media file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        File mediaFile;
        if (type == MEDIA_TYPE_IMAGE){
            mediaFile = new File(mediaStorageDir.getPath() + File.separator +
                    "IMG_"+ timeStamp + ".jpg");
            Log.d("MyCameraApp", mediaFile.toString());
        } else if(type == MEDIA_TYPE_VIDEO) {
            mediaFile = new File(mediaStorageDir.getPath() + File.separator +
                    "VID_"+ timeStamp + ".mp4");
        } else {
            return null;
        }

        return mediaFile;
    }

    protected void onPause() {
        super.onPause();
        mCamera.release();              // release the camera immediately on pause event
    }

}
