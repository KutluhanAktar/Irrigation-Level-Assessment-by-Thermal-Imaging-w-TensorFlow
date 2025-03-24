         /////////////////////////////////////////////  
        //     Irrigation Level Assessment by      //
       //      Thermal Imaging w/ TensorFlow      //
      //             ---------------             //
     //              (Wio Terminal)             //           
    //             by Kutluhan Aktar           // 
   //                                         //
  /////////////////////////////////////////////

//
// Collect irrigation level data by thermal imaging, build and train a neural network model, and run the model directly on Wio Terminal.
//
// For more information:
// https://www.theamplituhedron.com/projects/Irrigation_Level_Assessment_by_Thermal_Imaging_w_TensorFlow/
//
//
// Connections
// Wio Terminal :  
//                                MLX90641 Thermal Imaging Camera (16x12 w/ 110° FOV)
// Grove Connector  -------------- Grove Connector


// Include the required libraries.
#include <TFT_eSPI.h>
//////////////////////////////
// Add to avoid system errors.
#undef max 
#undef min
//////////////////////////////
#include <Wire.h>
#include "MLX90641_API.h"
#include "MLX9064X_I2C_Driver.h"

// Import the required TensorFlow modules.
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/version.h"

// Import the converted TensorFlow Lite model.
#include "irrigation_model.h"

// TFLite globals, used for compatibility with Arduino-style sketches:
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow arrays.
  constexpr int kTensorArenaSize = 15 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// Define the threshold value for the model outputs (results).
float threshold = 0.75;

// Define the irrigation level (class) names and color codes: 
String classes[] = {"Dry", "Moderate", "Sufficient", "Excessive"};
uint32_t class_color_codes[] = {TFT_GREEN, TFT_GREENYELLOW, TFT_ORANGE, TFT_RED};

// Define the TFT screen:
TFT_eSPI tft;

// Define the MLX90641 Thermal Imaging Camera settings:
const byte MLX90641_address = 0x33; // Default 7-bit unshifted address of the MLX90641.
#define TA_SHIFT 12 // Default shift value for MLX90641 in the open air.
uint16_t eeMLX90641[832];
float MLX90641To[192];
uint16_t MLX90641Frame[242];
paramsMLX90641 MLX90641;
int errorno = 0;

// Define the maximum and minimum temperature values:
uint16_t MinTemp = 21;
uint16_t MaxTemp = 45;

// Define the data holders:
byte red, green, blue;
float a, b, c, d;

void setup() {
  Serial.begin(115200);

  // Initialize the I2C clock for the MLX90641 Thermal Imaging Camera.
  Wire.begin();
  Wire.setClock(2000000); // Increase the I2C clock speed to 2M.  

  // Initiate the TFT screen:
  tft.begin();
  tft.setRotation(3);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_BLACK);
  tft.setTextSize(1);
  
  // Check the connection status between the MLX90641 Thermal Imaging Camera and Wio Terminal.
  if(isConnected() == false){
    tft.fillScreen(TFT_RED);
    tft.drawString("MLX90641 not detected at default I2C address!", 5, 10);
    tft.drawString("Please check wiring. Freezing.", 5, 25);
    while (1);
  }
  // Get the MLX90641 Thermal Imaging Camera parameters:
  int status;
  status = MLX90641_DumpEE(MLX90641_address, eeMLX90641);
  errorno = status;//MLX90641_CheckEEPROMValid(eeMLX90641);//eeMLX90641[10] & 0x0040;//
 
  if(status != 0){
    tft.fillScreen(TFT_RED);
    tft.drawString("Failed to load system parameters!", 5, 10);
    while(1);
  }
 
  status = MLX90641_ExtractParameters(eeMLX90641, &MLX90641);
  //errorno = status;
  if(status != 0){
    tft.fillScreen(TFT_RED);
    tft.drawString("Parameter extraction failed!", 5, 10);
    while(1);
  }

  // Once params are extracted, release eeMLX90641 array:
  MLX90641_SetRefreshRate(MLX90641_address, 0x05); // Set rate to 16Hz.

  // Get the cutoff points:
  Getabcd();

  // TensorFlow Lite Model settings:
  
  // Set up logging (will report to Serial, even within TFLite functions).
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure.
  model = tflite::GetModel(irrigation_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // 5-Way Switch
  pinMode(WIO_5S_PRESS, INPUT_PULLUP);

  delay(1000);
  tft.fillScreen(TFT_BLUE);
}

void loop(){
 get_and_display_data_from_MLX90641(64, 20, 12, 12);

 // Execute the TensorFlow Lite model to make predictions on the irrigation levels (classes).
 if(digitalRead(WIO_5S_PRESS) == LOW) run_inference_to_make_predictions();

}

void run_inference_to_make_predictions(){
    // Initiate the results screen.
    tft.fillScreen(TFT_PURPLE);
    
    // Copy values (thermal imaging array) to the input buffer (tensor):
    for(int i = 0; i < 192; i++){
      model_input->data.f[i] = MLX90641To[i];
    }

    // Run inference:
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on the given input.");
    }

    // Read predicted y values (irrigation classes) from the output buffer (tensor):
    for(int i = 0; i<4; i++){
      if(model_output->data.f[i] >= threshold){
        // Display the detection result (class).
        // Border:
        int w = 200; int h = 60;
        int x = (320 - w) / 2;
        int y = (240 - h) / 2;
        int offset = 15; int border = 1;
        int r = 20;
        tft.drawRoundRect(x - offset - border, y - offset - border, w + (2*offset) + (2*border), h + (2*offset) + (2*border), r, TFT_WHITE);
        tft.fillRoundRect(x - offset, y - offset, w + (2*offset), h + (2*offset), r, TFT_MAGENTA);
        tft.fillRoundRect(x, y, w, h, r, class_color_codes[i]);
        // Print:
        int str_x = classes[i].length() * 11;
        int str_y = 12;
        tft.setTextSize(2);
        tft.drawString(classes[i], (320 - str_x) / 2, (240 - str_y) / 2);
      }
    }
    
    // Exit and clear.
    delay(3000);
    tft.setTextSize(1);
    tft.fillScreen(TFT_BLUE);
}

void draw_menu(int start_x, int start_y, int w, int h){
  // Draw the border:
  int offset = 10;
  tft.drawRoundRect(start_x-offset, start_y-offset, (2*offset)+(w*16), (2*offset)+(h*12), 10, TFT_WHITE);
  // Draw options:
  int x_c = 320 / 2; int x_s = x_c - 7; int y_c = 210; int y_s = y_c - 11;
  tft.setTextSize(1);
  /////////////////////////////////////
  tft.fillCircle(x_c, y_c, 20, TFT_WHITE);
  tft.drawChar(x_s, y_s, 'P', TFT_BLACK, TFT_WHITE, 3);
  tft.drawString("'Press' to run the NN model", x_c - 80, y_c - 33);
  /////////////////////////////////////
}

void get_and_display_data_from_MLX90641(int start_x, int start_y, int w, int h){
  // Draw the options menu:
  draw_menu(start_x, start_y, w, h);
  // Elicit the 16x12 pixel IR thermal imaging array generated by the MLX90641 Thermal Imaging Camera:
  for(byte x = 0 ; x < 2 ; x++){
    int status = MLX90641_GetFrameData(MLX90641_address, MLX90641Frame);
    // Obtain the required variables to calculate the thermal imaging array:
    float vdd = MLX90641_GetVdd(MLX90641Frame, &MLX90641);
    float Ta = MLX90641_GetTa(MLX90641Frame, &MLX90641);
    // Define the reflected temperature based on the sensor's ambient temperature:
    float tr = Ta - TA_SHIFT; 
    float emissivity = 0.95;
    // Create the thermal imaging array:
    MLX90641_CalculateTo(MLX90641Frame, &MLX90641, emissivity, tr, MLX90641To);
  }
  // Define parameters: 
  int x = start_x;
  int y = start_y + (h*11);
  uint32_t c = TFT_BLUE; 
  for(int i = 0 ; i < 192 ; i++){
    // Display a simple image version of the collected data (array) on the screen:
    // Define the color palette:
    c = GetColor(MLX90641To[i]);
    // Draw image pixels (rectangles):
    tft.fillRect(x, y, w, h, c);
    x = x + w;
    // Start a new row:
    int l = i + 1;
    if (l%16 == 0) { x = start_x; y = y - h; }
  }
}

boolean isConnected(){
  // Return true if the MLX90641 Thermal Imaging Camera is detected on the I2C bus:
  Wire.beginTransmission((uint8_t)MLX90641_address);
  if(Wire.endTransmission() != 0){
    return (false);
  }
  return (true);
}

uint16_t GetColor(float val){
  /*
 
    equations based on
    http://web-tech.ga-usa.com/2012/05/creating-a-custom-hot-to-cold-temperature-color-gradient-for-use-with-rrdtool/index.html
 
  */

  // Assign colors to the given temperature readings:
  // R:
  red = constrain(255.0 / (c - b) * val - ((b * 255.0) / (c - b)), 0, 255);
  // G:
  if((val > MinTemp) & (val < a)){
    green = constrain(255.0 / (a - MinTemp) * val - (255.0 * MinTemp) / (a - MinTemp), 0, 255);
  }else if((val >= a) & (val <= c)){
    green = 255;
  }else if(val > c){
    green = constrain(255.0 / (c - d) * val - (d * 255.0) / (c - d), 0, 255);
  }else if((val > d) | (val < a)){
    green = 0;
  }
  // B:
  if(val <= b){
    blue = constrain(255.0 / (a - b) * val - (255.0 * b) / (a - b), 0, 255);
  }else if((val > b) & (val <= d)){
    blue = 0;
  }else if (val > d){
    blue = constrain(240.0 / (MaxTemp - d) * val - (d * 240.0) / (MaxTemp - d), 0, 240);
  }
 
  // Utilize the built-in color mapping function to get a 5-6-5 color palette (R=5 bits, G=6 bits, B-5 bits):
  return tft.color565(red, green, blue);
}

void Getabcd() {
  // Get the cutoff points based on the given maximum and minimum temperature values.
  a = MinTemp + (MaxTemp - MinTemp) * 0.2121;
  b = MinTemp + (MaxTemp - MinTemp) * 0.3182;
  c = MinTemp + (MaxTemp - MinTemp) * 0.4242;
  d = MinTemp + (MaxTemp - MinTemp) * 0.8182;
}
