package com.example.objectdetection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.objectdetection.ml.EfficientdetLite1
import org.tensorflow.lite.support.image.TensorImage

class MainActivity : AppCompatActivity() {

    private val paint = Paint()
    private lateinit var textureView: TextureView
    private lateinit var imageView: ImageView
    private lateinit var captureButton: Button
    private lateinit var cameraDevice: CameraDevice
    private lateinit var handler: Handler
    private lateinit var cameraManager: CameraManager
    private var isImageCaptured = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        getPermission()

        paint.textSize = 40f
        paint.strokeWidth = 5f

        textureView = findViewById(R.id.textureView)
        imageView = findViewById(R.id.imageView)
        captureButton = findViewById(R.id.captureButton)

        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(
                surface: SurfaceTexture,
                width: Int,
                height: Int
            ) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
        }

        captureButton.setOnClickListener {
            if (isImageCaptured) {
                resetCapture()
            } else {
                captureAndProcessImage()
            }
        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        cameraManager.openCamera(cameraManager.cameraIdList[0], object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                val surfaceTexture = textureView.surfaceTexture
                val surface = Surface(surfaceTexture)

                val captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        session.setRepeatingRequest(captureRequest.build(), null, null)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {}
                }, handler)
            }

            override fun onDisconnected(camera: CameraDevice) {}

            override fun onError(camera: CameraDevice, error: Int) {}
        }, handler)
    }

    private fun captureAndProcessImage() {
        val bitmap = textureView.bitmap ?: return
        val model = EfficientdetLite1.newInstance(this)

        // Converts the bitmap to TensorImage format
        val image = TensorImage.fromBitmap(bitmap)

        // Run inference
        val outputs = model.process(image)
        val detectionResults = outputs.detectionResultList

        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)


        val h = mutableBitmap.height
        val w = mutableBitmap.width

        paint.textSize = h / 20f // Scale text size based on image height
        paint.strokeWidth = h / 150f

        for (detectionResult in detectionResults) {
            val location = detectionResult.locationAsRectF
            val category = detectionResult.categoryAsString
            val score = detectionResult.scoreAsFloat

            if (score > 0.3) { // Reduced confidence threshold to see more predictions
                // Draw bounding box
                paint.color = Color.RED
                paint.style = Paint.Style.STROKE
                canvas.drawRect(location, paint)

                // Draw label background
                paint.style = Paint.Style.FILL
                paint.color = Color.argb(150, 0, 0, 0) // Semi-transparent black
                val label = "$category (${String.format("%.2f", score)})"
                val labelWidth = paint.measureText(label)
                val labelHeight = paint.textSize
                val labelBackground = RectF(
                    location.left,
                    location.top - labelHeight - 10,
                    location.left + labelWidth + 10,
                    location.top
                )
                canvas.drawRect(labelBackground, paint)

                // Adjust label position if it goes out of frame
                paint.color = Color.WHITE
                val adjustedX = if (location.left + labelWidth > w) w - labelWidth - 10 else location.left
                val adjustedY = if (location.top < labelHeight) location.top + labelHeight + 10 else location.top
                canvas.drawText(label, adjustedX, adjustedY, paint)
            }
        }

        model.close()

        // Display the result in the ImageView
        imageView.setImageBitmap(mutableBitmap)
        imageView.visibility = ImageView.VISIBLE
        textureView.visibility = TextureView.GONE

        // Update the button text
        captureButton.text = "Recapture"
        isImageCaptured = true
    }

    private fun resetCapture() {
        imageView.visibility = ImageView.GONE
        textureView.visibility = TextureView.VISIBLE
        captureButton.text = "Capture"
        isImageCaptured = false
    }

    private fun getPermission() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getPermission()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraDevice.close()
    }
}
