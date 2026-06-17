// Defines the package namespace for this file
package com.example.weightpredictor

// Android Framework imports for UI context, lifecycle, logs, and user alerts
import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.Toast

// Jetpack Compose activity and UI hierarchy bootstrap imports
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background

// Jetpack Compose layout structure imports (Box, Column, Row, Spacer, etc.)
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions

// Material Design 3 UI components and styling imports
import androidx.compose.material3.*

// Compose State management hooks for tracking data changes and triggering recompositions
import androidx.compose.runtime.*

// Layout alignment options and modifiers for customizing UI styling
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp

// Kotlin Coroutines for asynchronous, non-blocking background operations
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter

// Core Java/Kotlin utilities for memory buffer handling and cryptographic hashing
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

class MainActivity : ComponentActivity() {
    // Reference to our custom TensorFlow Lite model wrapper, initially null
    private var helper: TFLiteHelper? = null
    // Creates a CoroutineScope bound to the main thread lifecycle for UI operations
    private val scope = MainScope()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Immediately attaches and renders the Jetpack Compose UI layout
        setContent {
            // Applies global Material Design 3 styling rules (colors, typography)
            MaterialTheme {
                // Background surface that expands to fill the device screen
                Surface(Modifier.fillMaxSize()) {
                    // Renders the main screen UI and provides a lambda expression to run inference
                    PredictorScreen(
                        // If helper is initialized, it calls predictWeightKg; otherwise returns NaN
                        onPredict = { h -> helper?.predictWeightKg(h) ?: Float.NaN }
                    )
                }
            }
        }

        // Offloads heavy file I/O operations and model setup to a background thread
        scope.launch(Dispatchers.Default) {
            // Safely attempts to instantiate TFLiteHelper without crashing if an error occurs
            val result = runCatching {
                TFLiteHelper(this@MainActivity)
            }
            // Switches back to the Main UI Thread to update the app's state safely
            withContext(Dispatchers.Main) {
                // If model loading succeeds, save the instance and notify the user
                result.onSuccess { h ->
                    helper = h
                    Toast.makeText(this@MainActivity, "Model is ready", Toast.LENGTH_SHORT).show()
                }.onFailure {
                    Log.e("TFLite", "Model load failed", it)
                    Toast.makeText(this@MainActivity, "Model failed to load", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
    // Called automatically right before the activity is destroyed
    override fun onDestroy(){
        super.onDestroy()
        // Gracefully frees native memory allocated by the TensorFlow Lite interpreter
        runCatching { helper?.close() }
        // Cancels any pending background operations/coroutines to prevent memory leaks
        scope.cancel()
    }
}

// Wrapper class handling the heavy lifting of the TensorFlow Lite Model lifecycle
class TFLiteHelper(context :Context){
    // The core TFLite instance that executes the model's graph
    private val interpreter : Interpreter
    // Initializer block executed when TFLiteHelper is instantiated
    init{
        val assetName = "height_weight.tflite"
        val bytes = context.assets.open(assetName).use{
            it.readBytes()
        }
        // Prints debug metadata to verification logs (file size and signature hash)
        Log.d("TFLite", "asset =$assetName size=${bytes.size} shal6=${shal6(bytes)}")

        // Allocates direct native memory outside JVM heap (required for high-performance TFLite inference)
        val bb = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        // Loads the asset bytes into the buffer and resets the buffer pointer to position zero
        bb.put(bytes).rewind()

        // Configuration settings for optimizing TFLite model execution
        val opts = Interpreter.Options().apply{
            setUseXNNPACK(false)
            setNumThreads(1)
        }

        // Instantiates the TFLite runtime using our direct byte buffer and configuration options
        interpreter = Interpreter(bb,opts)

        // Local helper function to extract shape and data type from a model's IO Tensors
        fun info(t: org.tensorflow.lite.Tensor) = "shape${t.shape().contentToString()} type=${t.dataType()}"

        // Logs input/output structure details to verify the expected input/output shapes match our code
        Log.d("TFLITE","INPUT -> ${info(interpreter.getInputTensor(0))}")
        Log.d("TFLITE","OUTPUT -> ${info(interpreter.getOutputTensor(0))}")
    }

    // Generates a short string hash to uniquely identify and track model files
    private fun shal6 (b: ByteArray):String {
        val md = MessageDigest.getInstance("SHA-256").digest(b)
        return md.take(8).joinToString(""){
            "%02x".format(it)
        }
    }

    // Primary internal function dealing with structural variations of the TFLite file
    private fun safePredict (h: Float): Float {
        val int = interpreter.getInputTensor(0)
        val out = interpreter.getOutputTensor(0)

        // Enforces safety guardrails
        require(int.dataType().name == "FLOAT32" && out.dataType().name == "FLOAT32") {
            "Expected FLOAT32 model got in=${int.dataType()} out=${out.dataType()}"
        }

        // Evaluates input layer dimensions to adapt and structuralize Java array wrapper
        val inputObj: Any = when (int.shape().size) {
            1 -> floatArrayOf(h) //[1]
            2 -> arrayOf(floatArrayOf(h)) //[1,1]
            else -> error("Unsupported input shape ${int.shape().contentToString()}")
        }

        // Dynamically creates output container structure following equivalent dimensions
        val outputObj: Any = when (out.shape().size) {
            1 -> FloatArray(1)
            2 -> arrayOf(FloatArray(1))
            else -> error("Unsupported output shape ${out.shape().contentToString()}")
        }

        // Triggers the underlying native C++ engine execution
        interpreter.run(inputObj, outputObj)

        // Type-checks the generic Java Object matrix back into usable primitive types
        val y = when (outputObj) {
            is FloatArray -> outputObj[0]
            is Array<*> -> (outputObj[0] as FloatArray)[0]
            else -> Float.NaN
        }

        return if (y.isFinite()) y else Float.NaN
    }

    // Public safe execution API facing external activity components
    fun predictWeightKg(heightCm:Float) : Float = try {
        val y = safePredict(heightCm)
        Log.d("TFLite","predict h=$heightCm -> $y")
        y
    } catch (t: Throwable){
        Log.e("TFLite", "predict failed", t)
        Float.NaN
    }

    // Explicit clean up of allocated C++ runtime addresses
    fun close() = interpreter.close()
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PredictorScreen(onPredict: (Float) -> Float) {

    var heightText by remember { mutableStateOf("") }
    var result by remember { mutableStateOf<Float?>(null) }

    val Background = Color(0xFFF5FAFC)

    val HeaderGradient = Brush.verticalGradient(
        colors = listOf(
            Color(0xFF00B4D8),
            Color(0xFF0077B6)
        )
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Background)
    ) {

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(250.dp)
                .background(HeaderGradient)
        )

        Card(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 180.dp),

            shape = RoundedCornerShape(
                topStart = 40.dp,
                topEnd = 40.dp
            ),

            colors = CardDefaults.cardColors(
                containerColor = Color.White
            )
        ) {

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),

                horizontalAlignment = Alignment.Start
            ) {
                Text(
                    text = "Weight Predictor",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Enter your height and predict weight using the TFLite model.",
                    color = Color.Gray
                )

                Spacer(modifier = Modifier.height(32.dp))

                OutlinedTextField(
                    value = heightText,

                    onValueChange = {
                        heightText = it
                    },

                    label = {
                        Text("Height (cm)")
                    },

                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Number
                    ),

                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(24.dp))

                Button(
                    onClick = {
                        val height = heightText.toFloatOrNull()
                        result = height?.let {
                            onPredict(it)
                        }
                    },

                    modifier = Modifier
                        .fillMaxWidth()
                        .height(55.dp),

                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF0077B6)
                    )
                ) {
                    Text(
                        text = "PREDICT",
                        color = Color.White
                    )
                }

                Spacer(modifier = Modifier.height(30.dp))

                Card(
                    modifier = Modifier.fillMaxWidth(),

                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFFF7F9FA)
                    )
                ) {

                    Text(
                        text = when {
                            result == null ->
                                "Enter a height and press Predict."

                            result!!.isNaN() ->
                                "Prediction failed."

                            else ->
                                "Predicted weight: ${"%.1f".format(result)} kg"
                        },

                        modifier = Modifier.padding(20.dp),

                        style = MaterialTheme.typography.bodyLarge
                    )
                }
            }
        }
    }
}