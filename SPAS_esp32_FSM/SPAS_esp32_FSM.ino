#include <Arduino.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include <esp_now.h>
#include <SPI.h>
#include "MS5607.h"
#include <esp_timer.h>

// Pin setting
#define IMU1_RX 22
#define IMU1_TX 27
#define IMU2_RX 21
#define IMU2_TX 32
#define IMU3_RX 16
#define IMU3_TX 17
#define PMMG_CS 5
#define BUTTON 26
#define LED_PIN 33
#define BATTERY 36

// FSM states
enum State {
  IDLE,
  LOW_VOLTAGE,
  PMMG_ERROR,
  IMUS_ERROR,
  STANDBY,
  LEG_ZEROING,
  LEG_ZEROING_END,
  READ_SENSORS,
  READ_SENSORS_END,
  MAG_CALIBRATION
};

enum MagCalState {
  START_CALIBRATION,
  WAIT_FOR_BUTTON_PRESS,
  SEND_CALIBRATION_COMMANDS
};

volatile State currentState = IDLE;
volatile State nextState = IDLE;
volatile MagCalState magCalibrationState = START_CALIBRATION;

#define MSG_STATE_IS_STANDBY 101
#define MSG_ZEROING_STARTED 102
#define MSG_ZEROING_STOPPED 103
#define MSG_READING_STARTED 104
#define MSG_READING_STOPPED 105
#define MSG_STATE_IS_MAG_CAL 106
#define ERROR_LOW_VOLTAGE 201
#define ERROR_PMMG_MALFUNC 202
#define ERROR_IMUS_MALFUNC 203

uint8_t stateMessage = 0;

struct Quaternion {
  float qw, qx, qy, qz;
  bool valid;  // Indicator for data validity
};
Quaternion quat_imu1, quat_imu2, quat_imu3;

HardwareSerial IMUSerial1(0);
HardwareSerial IMUSerial2(1);
HardwareSerial IMUSerial3(2);

MS5607 PMMG(PMMG_CS, 0);
volatile bool processDataFlag = false;  // Flag to trigger process data

volatile bool button_now = 1, button_old = 1, button_is_pushed = 0;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;
const uint16_t battery_threshold = 4000;

#define LONG_PRESS_TIME 1000
volatile unsigned long stateEntryTime = 0;  // Stores the time when a specific state is entered


// Declare a handle for the timer
void IRAM_ATTR onTimer(void* arg);
void IRAM_ATTR led_blink_timer_callback(void* arg);
esp_timer_handle_t mainTimer = NULL;
esp_timer_handle_t ledBlinkTimer = NULL;

// WI-FI
uint8_t broadcastAddress[] = { 0xAC, 0x67, 0xB2, 0x40, 0x13, 0x8C };
esp_now_peer_info_t peerInfo;

struct struct_message {
  uint8_t msg;
  unsigned long time;
  int16_t d_imu1[4];  // IMU1 quaternion data in fixed-point format
  int16_t d_imu2[4];  // IMU2 quaternion data in fixed-point format
  int16_t d_imu3[4];  // IMU3 quaternion data in fixed-point format
  float d_pmmg;
} myData;

const float SCALE_FACTOR = 1000.0;


// ################################################################# //
void setup() {
  pinMode(BUTTON, INPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BATTERY, INPUT);
  pinMode(PMMG_CS, OUTPUT);

  digitalWrite(LED_PIN, HIGH);

  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    // Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_send_cb(OnDataSent);
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    return;
  }

  const esp_timer_create_args_t maintimerArgs = {
    .callback = &onTimer,
    .arg = NULL,
    .name = "periodicTimer"
  };

  const esp_timer_create_args_t ledBlinkTimerArgs = {
    .callback = &led_blink_timer_callback,
    .arg = NULL,
    .name = "ledBlinkTimer"
  };

  esp_timer_create(&maintimerArgs, &mainTimer);
  esp_timer_create(&ledBlinkTimerArgs, &ledBlinkTimer);
  esp_timer_start_periodic(mainTimer, 5000);
}
// ################################################################# //

// ################################################################# //
//
//                   ███████╗ ██████╗███╗   ███╗
//                   ██╔════╝██╔════╝████╗ ████║
//                   █████╗  ╚█████╗ ██╔████╔██║
//                   ██╔══╝   ╚═══██╗██║╚██╔╝██║
//                   ██║     ██████╔╝██║ ╚═╝ ██║
//                   ╚═╝     ╚═════╝ ╚═╝     ╚═╝

void loop() {
  // Step 1: Button Checking
  button_is_pushed = button_check();

  // Step 2: State Handling
  switch (currentState) {
    case IDLE: handleIdle(); break;
    case LOW_VOLTAGE: handleLowVoltage(); break;
    case PMMG_ERROR: handlepmmgError(); break;
    case IMUS_ERROR: handleimusError(); break;
    case STANDBY: handleStandby(); break;
    case LEG_ZEROING: handleLegZeroing(); break;
    case LEG_ZEROING_END: handleLegZeroingEnd(); break;
    case READ_SENSORS: handleReadSensors(); break;
    case READ_SENSORS_END: handleReadSensorsEnd(); break;
    case MAG_CALIBRATION: handleMagCalibration(); break;
  }

  // Step 3: Sensor Data Processing (conditional on processDataFlag)
  if (processDataFlag) {
    processDataFlag = false;

    // 1) pMMG 읽기
    bool pmmgSuccess = (PMMG.read(8) == 0);

    if (pmmgSuccess) {
      // 2) pMMG 성공 시 IMU pooling 트리거
      IMUSerial1.write('*');
      IMUSerial2.write('*');
      IMUSerial3.write('*');
    }

    // 3) 각 IMU의 pooling 응답을 읽어서 quaternion 업데이트
    bool imu1Success = readPooledQuaternion(IMUSerial1, quat_imu1);
    bool imu2Success = readPooledQuaternion(IMUSerial2, quat_imu2);
    bool imu3Success = readPooledQuaternion(IMUSerial3, quat_imu3);

    // 4) 전체 성공 판정
    bool newData = pmmgSuccess && imu1Success && imu2Success && imu3Success;

    if (!pmmgSuccess || PMMG.getPressure() * 0.001 > 150) {
      nextState = PMMG_ERROR;
      storeMsg(ERROR_PMMG_MALFUNC);
    } else if (!newData) {
      nextState = IMUS_ERROR;
      storeMsg(ERROR_IMUS_MALFUNC);
    } else {
      // 5) 데이터 저장·전송
      myData.d_pmmg  = PMMG.getPressure() * 0.001;
      storeData();
      sendData_espnow();
      //}
    }
  }

  // Step 4: State Transition
  if (nextState != currentState) {
    currentState = nextState;
    if (nextState == LEG_ZEROING_END || nextState == READ_SENSORS_END) {
      // Optional delay to ensure data is sent before state transition
      delay(200);
    }
    sendData_espnow();  // Send state change message
  }
}

void handleIdle() {
  esp_timer_stop(mainTimer);

  IMUSerial1.begin(921600, SERIAL_8N1, IMU1_RX, IMU1_TX);
  IMUSerial2.begin(921600, SERIAL_8N1, IMU2_RX, IMU2_TX);
  IMUSerial3.begin(921600, SERIAL_8N1, IMU3_RX, IMU3_TX);
  SPI.begin();
  PMMG.init();

  esp_timer_start_periodic(mainTimer, 5000);

  nextState = (analogRead(BATTERY) < battery_threshold) ? LOW_VOLTAGE : STANDBY;
  storeMsg(nextState == LOW_VOLTAGE ? ERROR_LOW_VOLTAGE : MSG_STATE_IS_STANDBY);
}

void handleLowVoltage() {}

void handlepmmgError() {
  if (button_is_pushed) {
    nextState = IDLE;
  }
}

void handleimusError() {
  if (button_is_pushed) {
    nextState = IDLE;
  }
}

void handleStandby() {
  static unsigned long buttonPressTime = 0;

  if (button_is_pushed) {
    buttonPressTime = millis();
  } else if (buttonPressTime > 0 && (millis() - buttonPressTime) > LONG_PRESS_TIME) {
    nextState = MAG_CALIBRATION;
    magCalibrationState = START_CALIBRATION;
    storeMsg(MSG_STATE_IS_MAG_CAL);
    buttonPressTime = 0;
  } else if (buttonPressTime > 0 && (millis() - buttonPressTime) < LONG_PRESS_TIME) {
    if (digitalRead(BUTTON) == HIGH) {
      nextState = LEG_ZEROING;
      stateEntryTime = millis();
      digitalWrite(LED_PIN, LOW);
      storeMsg(MSG_ZEROING_STARTED);
      buttonPressTime = 0;
    }
  }
}

void handleLegZeroing() {
  if (button_is_pushed) {
    digitalWrite(LED_PIN, HIGH);
    nextState = LEG_ZEROING_END;
    storeMsg(MSG_ZEROING_STOPPED);
  }
}

void handleLegZeroingEnd() {
  if (button_is_pushed) {
    nextState = READ_SENSORS;
    esp_timer_start_periodic(ledBlinkTimer, 100000);
    stateEntryTime = millis();
    storeMsg(MSG_READING_STARTED);
  }
}

void handleReadSensors() {
  if (button_is_pushed) {
    nextState = READ_SENSORS_END;
    esp_timer_stop(ledBlinkTimer);
    digitalWrite(LED_PIN, HIGH);
    storeMsg(MSG_READING_STOPPED);
  }
}

void handleReadSensorsEnd() {
  static unsigned long buttonPressTime = 0;

  if (button_is_pushed) {
    buttonPressTime = millis();
  } else if (buttonPressTime > 0 && (millis() - buttonPressTime) > LONG_PRESS_TIME) {
    nextState = STANDBY;
    storeMsg(MSG_STATE_IS_STANDBY);
    buttonPressTime = 0;
  } else if (buttonPressTime > 0 && (millis() - buttonPressTime) < LONG_PRESS_TIME) {
    if (digitalRead(BUTTON) == HIGH) {
      nextState = READ_SENSORS;
      esp_timer_start_periodic(ledBlinkTimer, 100000);
      stateEntryTime = millis();
      storeMsg(MSG_READING_STARTED);
      buttonPressTime = 0;
    }
  }
}


// void handleMagCalibration() {
//   IMUSerial1.println("<cmf>");
//   IMUSerial2.println("<cmf>");
//   IMUSerial3.println("<cmf>");

//   while (!button_check()) { delay(10); }

//   bool magcal_allok = true;

//   IMUSerial1.println(">");
//   magcal_allok &= waitForCalibrationResponse(IMUSerial1);
//   IMUSerial2.println(">");
//   magcal_allok &= waitForCalibrationResponse(IMUSerial2);
//   IMUSerial3.println(">");
//   magcal_allok &= waitForCalibrationResponse(IMUSerial3);

//   if (magcal_allok) {
//     nextState = STANDBY;
//     storeMsg(MSG_STATE_IS_STANDBY);
//   } else {
//     nextState = IMUS_ERROR;
//     storeMsg(ERROR_IMUS_MALFUNC);
//   }
// }

void handleMagCalibration() {
  switch (magCalibrationState) {
    case START_CALIBRATION:
      IMUSerial1.println("<cmf>");
      IMUSerial2.println("<cmf>");
      IMUSerial3.println("<cmf>");
      magCalibrationState = WAIT_FOR_BUTTON_PRESS;
      break;

    case WAIT_FOR_BUTTON_PRESS:
      if (button_check()) {
        magCalibrationState = SEND_CALIBRATION_COMMANDS;
      }
      break;

    case SEND_CALIBRATION_COMMANDS:
      bool magcal_allok = true;
      IMUSerial1.println(">");
      magcal_allok &= waitForCalibrationResponse(IMUSerial1);
      IMUSerial2.println(">");
      magcal_allok &= waitForCalibrationResponse(IMUSerial2);
      IMUSerial3.println(">");
      magcal_allok &= waitForCalibrationResponse(IMUSerial3);
      if (magcal_allok) {
        nextState = STANDBY;
        storeMsg(MSG_STATE_IS_STANDBY);
      } else {
        nextState = IMUS_ERROR;
        storeMsg(ERROR_IMUS_MALFUNC);
      }
      break;
  }
}

// bool waitForCalibrationResponse(HardwareSerial& serial) {
//   unsigned long timeout = millis() + 1000;
//   while (millis() < timeout) {
//     if (serial.available()) {
//       String response = serial.readStringUntil('\n');
//       if (response == "<ok>") {
//         return true;
//       }
//     }
//   }
//   return false;
// }

bool waitForCalibrationResponse(HardwareSerial& serial) {
  unsigned long startTime = millis();
  unsigned long timeout = 1000; // 타임아웃 시간 설정 (1초)
  String response = "";

  while (millis() - startTime < timeout) {
    while (serial.available()) {
      char c = serial.read();
      response += c;

      // 버퍼 내에서 <ok> 문자열 검색
      if (response.indexOf("<ok>") != -1) {
        return true;
      }

      // 버퍼의 크기를 제한하여 메모리 사용을 최적화
      if (response.length() > 40) {
        response = response.substring(response.length() - 20); // 최근 20글자만 유지
      }
    }
  }
  return false; // 타임아웃 시 false 반환
}

// ################################################################# //

// ################################################################# //
//
//               ██╗       ██╗██╗      ███████╗██╗
//               ██║  ██╗  ██║██║      ██╔════╝██║
//               ╚██╗████╗██╔╝██║█████╗█████╗  ██║
//                ████╔═████║ ██║╚════╝██╔══╝  ██║
//                ╚██╔╝ ╚██╔╝ ██║      ██║     ██║
//                 ╚═╝   ╚═╝  ╚═╝      ╚═╝     ╚═╝

void sendData_espnow() {
  esp_err_t result = esp_now_send(broadcastAddress, (uint8_t*)&myData, sizeof(myData));
  if (result != ESP_OK) {
    // Handle error if needed
  }
}

void storeMsg(uint8_t messageCode) {
  myData.msg = messageCode;
  myData.time = millis();
  memset(myData.d_imu1, 0, sizeof(myData.d_imu1));
  memset(myData.d_imu2, 0, sizeof(myData.d_imu2));
  memset(myData.d_imu3, 0, sizeof(myData.d_imu3));
  myData.d_pmmg = 0;
}

void storeData() {
  unsigned long elapsedTime = millis() - stateEntryTime;  // Calculate elapsed time
  myData.msg = 0;
  myData.time = elapsedTime;
  myData.d_imu1[0] = (int16_t)(quat_imu1.qw * SCALE_FACTOR);
  myData.d_imu1[1] = (int16_t)(quat_imu1.qx * SCALE_FACTOR);
  myData.d_imu1[2] = (int16_t)(quat_imu1.qy * SCALE_FACTOR);
  myData.d_imu1[3] = (int16_t)(quat_imu1.qz * SCALE_FACTOR);

  myData.d_imu2[0] = (int16_t)(quat_imu2.qw * SCALE_FACTOR);
  myData.d_imu2[1] = (int16_t)(quat_imu2.qx * SCALE_FACTOR);
  myData.d_imu2[2] = (int16_t)(quat_imu2.qy * SCALE_FACTOR);
  myData.d_imu2[3] = (int16_t)(quat_imu2.qz * SCALE_FACTOR);

  myData.d_imu3[0] = (int16_t)(quat_imu3.qw * SCALE_FACTOR);
  myData.d_imu3[1] = (int16_t)(quat_imu3.qx * SCALE_FACTOR);
  myData.d_imu3[2] = (int16_t)(quat_imu3.qy * SCALE_FACTOR);
  myData.d_imu3[3] = (int16_t)(quat_imu3.qz * SCALE_FACTOR);
}

void OnDataSent(const uint8_t* mac_addr, esp_now_send_status_t status) {
  // Add any specific actions you need on successful or failed transmission
}
// ################################################################# //

// ################################################################# //
//          ██████╗███████╗███╗  ██╗ ██████╗ █████╗ ██████╗
//         ██╔════╝██╔════╝████╗ ██║██╔════╝██╔══██╗██╔══██╗
//         ╚█████╗ █████╗  ██╔██╗██║╚█████╗ ██║  ██║██████╔╝
//          ╚═══██╗██╔══╝  ██║╚████║ ╚═══██╗██║  ██║██╔══██╗
//         ██████╔╝███████╗██║ ╚███║██████╔╝╚█████╔╝██║  ██║
//         ╚═════╝ ╚══════╝╚═╝  ╚══╝╚═════╝  ╚════╝ ╚═╝  ╚═╝

void IRAM_ATTR onTimer(void* arg) {
  if (currentState == LEG_ZEROING || currentState == READ_SENSORS) {
    processDataFlag = true;
  }
}

void IRAM_ATTR led_blink_timer_callback(void* arg) {
  static int blinkState = 0;
  blinkState = (blinkState + 1) % 10;  // Cycle through 0 to 3

  // Define your blinking pattern
  if (blinkState == 0 || blinkState == 2) {
    digitalWrite(LED_PIN, LOW);
  } else {
    digitalWrite(LED_PIN, HIGH);
  }
}

bool updateQuaternion(HardwareSerial& serial, Quaternion& quat) {
  if (serial.available()) {
    Quaternion newQuat = readIMUSerial(serial);
    if (newQuat.valid) {
      quat = newQuat;
      return true;
    }
  }
  return true;
}

// '*' pooling 명령에 대한 응답을 기다려 quaternion을 읽어들임
bool readPooledQuaternion(HardwareSerial& serial, Quaternion& quat) {
  const unsigned long timeout_ms = 100;  // EBIMU 권장 pooling 응답 대기시간
  unsigned long start = millis();
  Quaternion newQ = {0,0,0,0,false};

  while (millis() - start < timeout_ms) {
    if (serial.available()) {
      newQ = readIMUSerial(serial);
      if (newQ.valid) {
        quat = newQ;
        return true;
      }
    }
  }
  return false;  // 타임아웃 시 실패
}

Quaternion readIMUSerial(HardwareSerial& serial) {
  static char buffer[64];
  static size_t bufPos = 0;
  Quaternion q = { 0.0, 0.0, 0.0, 0.0, false };

  while (serial.available()) {
    char c = serial.read();
    if (c == '\n' && bufPos > 0) {
      buffer[bufPos] = '\0';
      if (buffer[0] == '*' && bufPos > 1) {
        char* p = &buffer[1];
        float values[4];
        int count = 0;
        for (int i = 0; i < 4 && *p; i++) {
          values[i] = strtod(p, &p);
          if (p == &buffer[1] || (*p != ',' && *p != '\0')) {
            bufPos = 0;
            return q;
          }
          count++;
          if (*p == ',') p++;
        }
        if (count == 4) {
          q.qz = values[0];
          q.qy = values[1];
          q.qx = values[2];
          q.qw = values[3];
          q.valid = true;
        }
      }
      bufPos = 0;
      return q;
    } else if (c != '\r') {
      if (bufPos < sizeof(buffer) - 1) {
        buffer[bufPos++] = c;
      }
    }
  }
  return q;
}

bool button_check() {
  bool result = false;
  int reading = digitalRead(BUTTON);

  if (reading != button_old) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != button_now) {
      button_now = reading;
      if (button_now == LOW) {
        result = true;
      }
    }
  }

  button_old = reading;
  return result;
}
