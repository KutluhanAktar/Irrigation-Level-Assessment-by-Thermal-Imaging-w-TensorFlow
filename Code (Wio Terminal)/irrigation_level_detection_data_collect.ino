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
#include <rpcWiFi.h>
#include <TFT_eSPI.h>
#include <Wire.h>
#include "MLX90641_API.h"
#include "MLX9064X_I2C_Driver.h"

// Define the Wi-Fi network settings:
const char* ssid = "<_SSID_>";
const char* password =  "<_PASSWORD_>";

// Define the server settings:
const uint16_t port = 80; // Default port
const char* host = "192.168.1.20";  // Target Server IP Address

// Use the WiFiClient class to create TCP connections:
WiFiClient client;

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
String MLX90641_data = "";
byte red, green, blue;
float a, b, c, d;

void setup(){  
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
  
  // 5-Way Switch
  pinMode(WIO_5S_UP, INPUT_PULLUP);
  pinMode(WIO_5S_DOWN, INPUT_PULLUP);
  pinMode(WIO_5S_LEFT, INPUT_PULLUP);
  pinMode(WIO_5S_RIGHT, INPUT_PULLUP);
  pinMode(WIO_5S_PRESS, INPUT_PULLUP);

  // Set Wi-Fi to station mode and disconnect from an AP if it was previously connected:
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(2000);
 
  WiFi.begin(ssid, password);

  // Attempt to connect to the given Wi-Fi network:
  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    tft.fillScreen(TFT_RED);
    tft.drawString("Connecting to Wi-Fi...", 5, 10);
  }
  tft.setTextSize(2);
  tft.fillScreen(TFT_GREENYELLOW);
  tft.drawString("Connected to", 5, 10);
  tft.drawString("the Wi-Fi network!", 5, 40);
  delay(3000);
  tft.fillScreen(TFT_BLUE);

}
 
void loop(){  
  get_and_display_data_from_MLX90641(64, 20, 12, 12);
  
  // Define the data (CSV) files:
  if(digitalRead(WIO_5S_UP) == LOW) make_a_get_request("excessive");
  if(digitalRead(WIO_5S_LEFT) == LOW) make_a_get_request("sufficient");
  if(digitalRead(WIO_5S_RIGHT) == LOW) make_a_get_request("moderate");
  if(digitalRead(WIO_5S_DOWN) == LOW) make_a_get_request("dry");

}

void draw_menu(int start_x, int start_y, int w, int h){
  // Draw the border:
  int offset = 10;
  tft.drawRoundRect(start_x-offset, start_y-offset, (2*offset)+(w*16), (2*offset)+(h*12), 10, TFT_WHITE);
  // Draw options:
  int x_c = 52; int x_s = x_c - 7; int y_c = 210; int y_s = y_c - 11; int sp = 72;
  tft.setTextSize(1);
  /////////////////////////////////////
  tft.fillCircle(x_c, y_c, 20, TFT_WHITE);
  tft.drawChar(x_s, y_s, 'U', TFT_BLACK, TFT_WHITE, 3);
  tft.drawString("Excessive", x_c - 25, y_c - 33);
  /////////////////////////////////////
  tft.fillCircle(x_c + sp, y_c, 20, TFT_WHITE);
  tft.drawChar(x_s + sp, y_s, 'L', TFT_BLACK, TFT_WHITE, 3);
  tft.drawString("Sufficient", x_c + sp - 28, y_c - 33);
  /////////////////////////////////////
  tft.fillCircle(x_c + (2*sp), y_c, 20, TFT_WHITE);
  tft.drawChar(x_s + (2*sp), y_s, 'R', TFT_BLACK, TFT_WHITE, 3);
  tft.drawString("Moderate", x_s + (2*sp) - 16, y_c - 33);
  /////////////////////////////////////
  tft.fillCircle(x_c + (3*sp), y_c, 20, TFT_WHITE);
  tft.drawChar(x_s + (3*sp), y_s, 'D', TFT_BLACK, TFT_WHITE, 3);
  tft.drawString("Dry", x_c + (3*sp) - 8, y_c - 33);
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
  MLX90641_data = "";
  int x = start_x;
  int y = start_y + (h*11);
  uint32_t c = TFT_BLUE; 
  for(int i = 0 ; i < 192 ; i++){
    // Convert the 16x12 pixel IR thermal imaging array to string (MLX90641_data) so as to transfer it to the web application:
    MLX90641_data += String(MLX90641To[i]);
    if(i != 191) MLX90641_data += ",";
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

void make_a_get_request(String level){
  if(!client.connect(host, port)){
    tft.fillScreen(TFT_RED);
    tft.setTextSize(1);  
    tft.drawString("Connection failed!", 5, 10);
    tft.drawString("Waiting 5 seconds before retrying...", 5, 25);
    delay(5000);
    return;
  }
  // Make a Get request to the given server to send the recently generated IR thermal imaging array.
  String application = "/irrigation_level_data_logger/"; // Define the application name.
  String query = application + "?thermal_img=" + MLX90641_data + "&level=" + level;
  client.println("GET " + query + " HTTP/1.1");
  client.println("Host: 192.168.1.20");
  client.println("Connection: close");
  client.println();
  // Wait until the client is available.
  int maxloops = 0;
  while (!client.available() && maxloops < 2000) {
    maxloops++;
    delay(1);
  }
  // Fetch the response from the given application.
  if(client.available() > 0){
    String response = client.readString();
    if(response != "" && response.indexOf("The given line is added to the") > 0){
      tft.fillScreen(TFT_GREEN);
      tft.setTextSize(2);  
      tft.drawString("Data transferred", 5, 10);
      tft.drawString("successfully", 5, 40);
      tft.drawString("to the web application! ", 5, 70);
      tft.setTextSize(3); 
      tft.drawString(level + ".csv", 5, 130);
      
    }
  }else{    
    tft.fillScreen(TFT_RED);
    tft.setTextSize(2);  
    tft.drawString("Client Timeout!", 5, 10);
  }
  // Stop the client:
  client.stop();
  delay(3000);
  tft.fillScreen(TFT_BLUE);
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
