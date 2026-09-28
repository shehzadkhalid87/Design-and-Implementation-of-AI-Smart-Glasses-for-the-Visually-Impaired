/*
 * AI SMART GLASSES - ESP32-S3 + OV3660 + Edge Impulse FOMO
 *
 * Classes  : car, chair, door, person, stop sign
 * Alerts   : YX5300 voice track (/01/001..005) + buzzer beep pattern
 * Pipeline : OV3660 JPEG 320x240 -> RGB888 160x120 -> 96x96 -> FOMO
 * Buzzer   : GPIO14 (+) and GND (-)
 */

#include <Arduino.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <esp_jpg_decode.h>
#include <YX5300_ESP32.h>
#include <WiFi.h>
#include <WebServer.h>
#include <string.h>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "model-parameters/model_metadata.h"


// ================================================================
// CONFIGURATION
// ================================================================

#define ENABLE_AUDIO   1
#define ENABLE_BUZZER  1
#define BUZZER_ACTIVE  0   // 0 = passive buzzer (pitch), 1 = active buzzer (on/off only)

constexpr int      CAM_WIDTH  = 320, CAM_HEIGHT = 240;
constexpr int      SRC_WIDTH  = 160, SRC_HEIGHT = 120;   // JPEG decoded at 1/2 scale
constexpr int      MODEL_WIDTH  = EI_CLASSIFIER_INPUT_WIDTH;
constexpr int      MODEL_HEIGHT = EI_CLASSIFIER_INPUT_HEIGHT;

constexpr float    DETECTION_THRESHOLD    = 0.50f;
constexpr uint32_t ALERT_COOLDOWN_MS      = 3000;
constexpr uint32_t INFERENCE_INTERVAL_MS  = 250;
constexpr size_t   WEB_JPEG_MAX           = 128 * 1024;

constexpr size_t   MODEL_PIXELS = (size_t)MODEL_WIDTH * MODEL_HEIGHT;
constexpr size_t   SRC_BYTES    = (size_t)SRC_WIDTH * SRC_HEIGHT * 3;

#if EI_CLASSIFIER_OBJECT_DETECTION != 1
#error "This is not an Edge Impulse object detection model."
#endif
static_assert(EI_CLASSIFIER_INPUT_WIDTH == 96 && EI_CLASSIFIER_INPUT_HEIGHT == 96,
              "Expected 96x96 Edge Impulse model.");


// ================================================================
// GLOBALS
// ================================================================

static uint8_t  modelRgb[MODEL_PIXELS * 3];   // 96x96 RGB
static uint8_t *srcRgb  = nullptr;            // 160x120 RGB (PSRAM)
static uint8_t *webJpeg = nullptr;            // latest JPEG for the web page (PSRAM)
static size_t   webJpegLen = 0;
static SemaphoreHandle_t jpegMutex = nullptr;

static WebServer webServer(80);

struct Status {
    char     label[32];
    float    conf;
    uint32_t boxes, inferMs, frameMs;
    uint16_t x, y, w, h;
};
static Status status = {"none"};

struct CameraTiming { uint32_t grabUs, decodeUs, resizeUs; };


// ================================================================
// SQUASH RESIZE (RGB888, bilinear, matches Edge Impulse pipeline)
// ================================================================

static void squashRgb888(const uint8_t *src, int srcW, int srcH,
                         uint8_t *dst, int dstW, int dstH)
{
    constexpr int FRAC_BITS = 14;
    constexpr uint32_t FRAC_VAL  = 1u << FRAC_BITS;
    constexpr uint32_t FRAC_MASK = FRAC_VAL - 1;
    constexpr int PX = 3;

    const uint32_t xStep = ((uint32_t)srcW << FRAC_BITS) / dstW;
    const uint32_t yStep = ((uint32_t)srcH << FRAC_BITS) / dstH;
    const int rowBytes = srcW * PX;

    uint32_t yAcc = 0;
    for (int y = 0; y < dstH; y++, yAcc += yStep) {
        const uint8_t *s = src + (size_t)(yAcc >> FRAC_BITS) * rowBytes;
        const uint32_t yf = yAcc & FRAC_MASK, nyf = FRAC_VAL - yf;

        uint32_t xAcc = 0;
        for (int x = 0; x < dstW; x++, xAcc += xStep) {
            const uint8_t *p = s + (xAcc >> FRAC_BITS) * PX;
            const uint32_t xf = xAcc & FRAC_MASK, nxf = FRAC_VAL - xf;

            for (int c = 0; c < PX; c++) {
                uint32_t top = (p[c] * nxf + p[PX + c] * xf + FRAC_VAL / 2) >> FRAC_BITS;
                uint32_t bot = (p[rowBytes + c] * nxf + p[rowBytes + PX + c] * xf + FRAC_VAL / 2) >> FRAC_BITS;
                *dst++ = (uint8_t)((top * nyf + bot * yf + FRAC_VAL / 2) >> FRAC_BITS);
            }
        }
    }
}


// ================================================================
// CAMERA
// ================================================================

#define CAM_SET(fn, v) \
    do { if (sensor->fn(sensor, v)) Serial.println("[CAM] WARNING: " #fn " failed"); } while (0)

static bool beginCamera()
{
    Serial.println("\n[CAM] Initializing OV3660...");

    if (!psramFound()) {
        Serial.println("[CAM] ERROR: PSRAM not detected.");
        return false;
    }

    srcRgb  = (uint8_t *)heap_caps_malloc(SRC_BYTES,    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    webJpeg = (uint8_t *)heap_caps_malloc(WEB_JPEG_MAX, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!srcRgb || !webJpeg) {
        Serial.println("[CAM] ERROR: buffer allocation failed.");
        return false;
    }

    camera_config_t cfg = {};
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;

    cfg.pin_d0 = 11; cfg.pin_d1 = 9;  cfg.pin_d2 = 8;  cfg.pin_d3 = 10;
    cfg.pin_d4 = 12; cfg.pin_d5 = 18; cfg.pin_d6 = 17; cfg.pin_d7 = 16;
    cfg.pin_xclk  = 15; cfg.pin_pclk = 13;
    cfg.pin_vsync = 6;  cfg.pin_href = 7;
    cfg.pin_sccb_sda = 4; cfg.pin_sccb_scl = 5;
    cfg.pin_pwdn = -1;  cfg.pin_reset = -1;

    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_JPEG;
    cfg.frame_size   = FRAMESIZE_QVGA;
    cfg.jpeg_quality = 10;
    cfg.fb_count     = 2;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_LATEST;

    const esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        Serial.printf("[CAM] ERROR: esp_camera_init failed: 0x%x\n", (unsigned)err);
        return false;
    }

    sensor_t *sensor = esp_camera_sensor_get();
    if (!sensor) {
        Serial.println("[CAM] ERROR: sensor not found.");
        esp_camera_deinit();
        return false;
    }
    Serial.printf("[CAM] Sensor PID = 0x%04X\n", (unsigned)sensor->id.PID);

    CAM_SET(set_vflip, 1);
    CAM_SET(set_hmirror, 0);
    CAM_SET(set_brightness, 1);
    CAM_SET(set_contrast, 0);
    CAM_SET(set_saturation, 0);
    CAM_SET(set_whitebal, 1);
    CAM_SET(set_awb_gain, 1);
    CAM_SET(set_wb_mode, 0);
    CAM_SET(set_gain_ctrl, 1);
    CAM_SET(set_exposure_ctrl, 1);
    CAM_SET(set_colorbar, 0);

    // Let auto exposure / white balance settle.
    Serial.println("[CAM] Letting exposure/WB settle...");
    for (uint32_t start = millis(); millis() - start < 1500UL; delay(20)) {
        if (camera_fb_t *fb = esp_camera_fb_get()) {
            esp_camera_fb_return(fb);
        }
    }

    Serial.printf("[CAM] READY | PSRAM free=%u | heap free=%u\n",
                  (unsigned)ESP.getFreePsram(), (unsigned)ESP.getFreeHeap());
    return true;
}


// ================================================================
// JPEG DECODE CALLBACKS
// ================================================================

struct JpegContext {
    const uint8_t *src;
    size_t   len;
    uint8_t *dst;
    uint16_t width, height;
    bool     dimensionsOk;
};

static size_t jpegRead(void *arg, size_t index, uint8_t *buf, size_t len)
{
    auto *ctx = (JpegContext *)arg;
    if (!ctx || index >= ctx->len) return 0;
    if (len > ctx->len - index) len = ctx->len - index;
    if (buf) memcpy(buf, ctx->src + index, len);
    return len;
}

static bool jpegWrite(void *arg, uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint8_t *data)
{
    auto *ctx = (JpegContext *)arg;
    if (!ctx) return false;

    if (!data) {   // first callback reports the output size
        if (x == 0 && y == 0) ctx->dimensionsOk = (w == ctx->width && h == ctx->height);
        return ctx->dimensionsOk;
    }

    if (!ctx->dimensionsOk || (uint32_t)x + w > ctx->width || (uint32_t)y + h > ctx->height) {
        return false;
    }

    for (uint16_t row = 0; row < h; row++) {
        memcpy(ctx->dst + ((size_t)(y + row) * ctx->width + x) * 3,
               data + (size_t)row * w * 3,
               (size_t)w * 3);
    }
    return true;
}


// ================================================================
// CAPTURE + PREPROCESS
// ================================================================

static bool captureModel(CameraTiming &timing, bool prepareModel)
{
    timing = {};

    uint32_t t0 = micros();
    camera_fb_t *fb = esp_camera_fb_get();
    timing.grabUs = micros() - t0;

    if (!fb) {
        Serial.println("[CAM] ERROR: fb_get() failed.");
        return false;
    }

    if (fb->format != PIXFORMAT_JPEG || fb->width != CAM_WIDTH || fb->height != CAM_HEIGHT) {
        Serial.printf("[CAM] ERROR: format=%d size=%ux%u\n",
                      (int)fb->format, (unsigned)fb->width, (unsigned)fb->height);
        esp_camera_fb_return(fb);
        return false;
    }

    // Update the web frame only if the web task isn't sending it right now,
    // so a slow client can never stall the camera loop.
    if (fb->len <= WEB_JPEG_MAX && xSemaphoreTake(jpegMutex, 0) == pdTRUE) {
        memcpy(webJpeg, fb->buf, fb->len);
        webJpegLen = fb->len;
        status.frameMs = millis();
        xSemaphoreGive(jpegMutex);
    }

    if (!prepareModel) {
        esp_camera_fb_return(fb);
        return true;
    }

    // JPEG -> RGB888 at half resolution (faster than full decode)
    JpegContext ctx = {fb->buf, fb->len, srcRgb, SRC_WIDTH, SRC_HEIGHT, false};

    t0 = micros();
    const esp_err_t jpegErr = esp_jpg_decode(fb->len, JPG_SCALE_2X, jpegRead, jpegWrite, &ctx);
    timing.decodeUs = micros() - t0;

    esp_camera_fb_return(fb);

    if (jpegErr != ESP_OK || !ctx.dimensionsOk) {
        Serial.printf("[CAM] ERROR: JPEG decode failed: 0x%x\n", (unsigned)jpegErr);
        return false;
    }

    // 160x120 -> 96x96
    t0 = micros();
    squashRgb888(srcRgb, SRC_WIDTH, SRC_HEIGHT, modelRgb, MODEL_WIDTH, MODEL_HEIGHT);
    timing.resizeUs = micros() - t0;

    return true;
}


// ================================================================
// EDGE IMPULSE SIGNAL: one float per pixel, packed 0xRRGGBB
// ================================================================

static int getData(size_t offset, size_t length, float *out)
{
    if (!out || offset > MODEL_PIXELS || length > MODEL_PIXELS - offset) {
        return -1;
    }

    const uint8_t *p = modelRgb + offset * 3;
    for (size_t i = 0; i < length; i++, p += 3) {
        out[i] = (float)(((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2]);
    }
    return EIDSP_OK;
}


// ================================================================
// WEB SERVER (Wi-Fi AP: live snapshot + detection status)
// ================================================================

static void handleWebRoot()
{
    static const char page[] = R"HTML(
<!doctype html><html><head><meta name="viewport" content="width=device-width">
<title>AI Smart Glasses</title>
<style>
body{font-family:system-ui;background:#111;color:#eee;margin:16px}
img{width:min(100%,640px);border:2px solid #555}
pre{white-space:pre-wrap;background:#222;padding:12px;border-radius:8px}
</style></head><body>
<h2>ESP32-S3 Camera + FOMO</h2>
<img id="cam"><pre id="status">Connecting...</pre>
<script>
const img=document.getElementById('cam'), out=document.getElementById('status');
let imageBusy=false, statusBusy=false;
function updateImage(){
  if(imageBusy)return;
  imageBusy=true;
  const next=new Image();
  next.onload=()=>{img.src=next.src;imageBusy=false};
  next.onerror=()=>{imageBusy=false};
  next.src='/snapshot.jpg?t='+Date.now();
}
async function updateStatus(){
  if(statusBusy)return;
  statusBusy=true;
  try{out.textContent=JSON.stringify(await (await fetch('/status',{cache:'no-store'})).json(),null,2)}
  catch(e){out.textContent='Status unavailable: '+e}
  statusBusy=false;
}
setInterval(updateImage,500);
setInterval(updateStatus,500);
updateImage(); updateStatus();
</script></body></html>
)HTML";
    webServer.send(200, "text/html", page);
}

static void handleWebSnapshot()
{
    if (webJpegLen == 0) {
        webServer.send(503, "text/plain", "No camera frame available");
        return;
    }

    webServer.sendHeader("Cache-Control", "no-store");
    webServer.sendHeader("Connection", "close");

    xSemaphoreTake(jpegMutex, portMAX_DELAY);
    webServer.setContentLength(webJpegLen);
    webServer.send(200, "image/jpeg", "");
    WiFiClient client = webServer.client();
    client.write(webJpeg, webJpegLen);
    xSemaphoreGive(jpegMutex);
}

static void handleWebStatus()
{
    char json[256];
    snprintf(json, sizeof(json),
             "{\"detection\":\"%s\",\"confidence\":%.3f,\"boxes\":%lu,\"x\":%u,\"y\":%u,"
             "\"width\":%u,\"height\":%u,\"inference_ms\":%lu,\"frame_age_ms\":%lu}",
             status.label, (double)status.conf, (unsigned long)status.boxes,
             (unsigned)status.x, (unsigned)status.y, (unsigned)status.w, (unsigned)status.h,
             (unsigned long)status.inferMs, (unsigned long)(millis() - status.frameMs));
    webServer.send(200, "application/json", json);
}

static void webServerTask(void *)
{
    for (;;) {
        webServer.handleClient();
        vTaskDelay(pdMS_TO_TICKS(2));
    }
}

static void beginWebServer()
{
    const char *ssid = "AISmartGlasses";
    const char *password = "smartglass";

    WiFi.mode(WIFI_AP);
    if (!WiFi.softAP(ssid, password)) {
        Serial.println("[WEB] ERROR: access point failed.");
        return;
    }

    webServer.on("/", HTTP_GET, handleWebRoot);
    webServer.on("/snapshot.jpg", HTTP_GET, handleWebSnapshot);
    webServer.on("/status", HTTP_GET, handleWebStatus);
    webServer.begin();
    xTaskCreatePinnedToCore(webServerTask, "web-server", 8192, nullptr, 1, nullptr, 0);

    Serial.printf("[WEB] Wi-Fi '%s' password '%s' -> http://%s/\n",
                  ssid, password, WiFi.softAPIP().toString().c_str());
}


// ================================================================
// ALERTS: voice track + beep pattern per class
// ================================================================
//
// Beep patterns work without language:
//   stop sign = 4 fast high beeps   car    = 3 beeps   person = 2 beeps
//   door      = 1 long low beep     chair  = 1 short low beep
// ================================================================

struct Beep  { uint16_t freq, onMs, offMs; uint8_t count; };
struct Alert { const char *label; uint8_t track; Beep beep; };

static const Alert alerts[] = {
    {"car",       1, {1000, 150, 150, 3}},
    {"chair",     2, { 600, 150, 150, 1}},
    {"door",      3, { 500, 700,   0, 1}},
    {"person",    4, {1000, 150, 150, 2}},
    {"stop sign", 5, {2000,  80,  80, 4}},
};
constexpr size_t ALERT_COUNT = sizeof(alerts) / sizeof(alerts[0]);

static uint32_t lastAlertMs = 0;

static int findAlert(const char *label)
{
    for (size_t i = 0; label && i < ALERT_COUNT; i++) {
        if (strcmp(label, alerts[i].label) == 0) return (int)i;
    }
    return -1;
}


// ---------------- Buzzer (own task, never blocks inference) ----------------

#if ENABLE_BUZZER

constexpr int BUZZER_PIN = 14;
constexpr int BUZZER_CH  = 2;      // camera uses LEDC channel 0 / timer 0

static volatile int8_t beepRequest = -1;

static void buzz(uint16_t freq)
{
#if BUZZER_ACTIVE
    digitalWrite(BUZZER_PIN, freq ? HIGH : LOW);
#else
    ledcWriteTone(BUZZER_CH, freq);   // 0 = silence
#endif
}

static void buzzerTask(void *)
{
    for (;;) {
        const int8_t i = beepRequest;
        if (i < 0) {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        beepRequest = -1;

        const Beep &b = alerts[i].beep;
        for (uint8_t n = 0; n < b.count; n++) {
            buzz(b.freq);
            vTaskDelay(pdMS_TO_TICKS(b.onMs));
            buzz(0);
            vTaskDelay(pdMS_TO_TICKS(b.offMs));
        }
    }
}

static void beginBuzzer()
{
#if BUZZER_ACTIVE
    pinMode(BUZZER_PIN, OUTPUT);
#else
    ledcSetup(BUZZER_CH, 2000, 8);
    ledcAttachPin(BUZZER_PIN, BUZZER_CH);
#endif
    buzz(0);
    xTaskCreate(buzzerTask, "buzzer", 2048, nullptr, 1, nullptr);

    buzz(2000);                 // short boot chirp to confirm wiring
    delay(100);
    buzz(0);

    Serial.printf("[BUZZER] GPIO%d ready (%s)\n", BUZZER_PIN, BUZZER_ACTIVE ? "active" : "passive");
}

#endif


// ---------------- YX5300 voice ----------------

#if ENABLE_AUDIO

constexpr int     AUDIO_RX_PIN = 2, AUDIO_TX_PIN = 1;
constexpr uint8_t AUDIO_VOLUME = 25;
constexpr uint8_t AUDIO_FOLDER = 1;

static YX5300_ESP32 *player = nullptr;

static void beginAudio()
{
    Serial.println("\n[AUDIO] Initializing YX5300...");
    delay(1000);

    static YX5300_ESP32 module(Serial2, AUDIO_RX_PIN, AUDIO_TX_PIN);
    player = &module;

    delay(500);
    player->setVolume(AUDIO_VOLUME);
    player->unmute();
    delay(300);

    Serial.printf("[AUDIO] RX=GPIO%d TX=GPIO%d volume=%u | tracks=%d module volume=%d\n",
                  AUDIO_RX_PIN, AUDIO_TX_PIN, (unsigned)AUDIO_VOLUME,
                  player->queryTrackCount(), player->queryCurrentVolume());

    // Boot test: play the car track once.
    player->playTrackInFolder(alerts[0].track, AUDIO_FOLDER);
}

#endif


static void announce(size_t i)
{
    Serial.printf("[ALERT] %s\n", alerts[i].label);

#if ENABLE_AUDIO
    if (player) player->playTrackInFolder(alerts[i].track, AUDIO_FOLDER);
#endif
#if ENABLE_BUZZER
    beepRequest = (int8_t)i;
#endif

    lastAlertMs = millis();
}


// ================================================================
// SETUP
// ================================================================

static void printModelInfo()
{
    Serial.printf("\nModel: %dx%d | labels=%d | detection=%d | threshold=%.2f\n",
                  EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT,
                  EI_CLASSIFIER_LABEL_COUNT, EI_CLASSIFIER_OBJECT_DETECTION,
                  (double)EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD);

    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        Serial.printf("  [%d] %s\n", i, ei_classifier_inferencing_categories[i]);
    }
}

void setup()
{
    Serial.begin(115200);
    delay(2000);

    Serial.println("\n========================================");
    Serial.println("AI SMART GLASSES | ESP32-S3 + OV3660 + FOMO");
    Serial.println("========================================");

    printModelInfo();

    jpegMutex = xSemaphoreCreateMutex();

    if (!beginCamera()) {
        Serial.println("[BOOT] CAMERA FAILED.");
        while (true) delay(1000);
    }

    Serial.println("[EI] Initializing classifier...");
    run_classifier_init();
    Serial.println("[EI] Classifier initialized.");

    beginWebServer();

#if ENABLE_BUZZER
    beginBuzzer();
#endif
#if ENABLE_AUDIO
    beginAudio();
#endif

    Serial.println("\n[BOOT] SYSTEM READY.\n");
}


// ================================================================
// MAIN LOOP
// ================================================================

void loop()
{
    static uint32_t lastInferenceMs = 0, lastLogMs = 0;

    const bool infer = lastInferenceMs == 0 || millis() - lastInferenceMs >= INFERENCE_INTERVAL_MS;

    CameraTiming timing;
    if (!captureModel(timing, infer)) {
        delay(100);
        return;
    }
    if (!infer) {
        delay(1);
        return;
    }

    signal_t signal = {};
    signal.total_length = MODEL_PIXELS;
    signal.get_data = getData;

    ei_impulse_result_t result = {};

    const uint32_t t0 = millis();
    const EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    const uint32_t inferMs = millis() - t0;
    lastInferenceMs = millis();   // next inference is scheduled after this one finishes

    if (err != EI_IMPULSE_OK) {
        Serial.printf("[EI] ERROR: run_classifier() = %d\n", (int)err);
        delay(200);
        return;
    }

    if (millis() - lastLogMs >= 1000UL) {
        Serial.printf("[TIME] grab=%.1f decode=%.1f resize=%.1f | DSP=%lu NN=%lu total=%lu ms\n",
                      timing.grabUs / 1000.0, timing.decodeUs / 1000.0, timing.resizeUs / 1000.0,
                      (unsigned long)result.timing.dsp,
                      (unsigned long)result.timing.classification,
                      (unsigned long)inferMs);
        lastLogMs = millis();
    }

    // Strongest box only: one alert per inference.
    int best = -1;
    float bestValue = 0.0f;
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        const auto &bb = result.bounding_boxes[i];
        if (bb.label && bb.value > bestValue) {
            bestValue = bb.value;
            best = (int)i;
        }
    }

    const auto *bb = best >= 0 ? &result.bounding_boxes[best] : nullptr;
    snprintf(status.label, sizeof(status.label), "%s", bb ? bb->label : "none");
    status.conf    = bb ? bb->value : 0.0f;
    status.boxes   = result.bounding_boxes_count;
    status.x       = bb ? bb->x : 0;
    status.y       = bb ? bb->y : 0;
    status.w       = bb ? bb->width : 0;
    status.h       = bb ? bb->height : 0;
    status.inferMs = inferMs;

    if (!bb || bb->value < DETECTION_THRESHOLD) return;

    Serial.printf("[DETECT] %s %.3f\n", bb->label, (double)bb->value);

    const int alert = findAlert(bb->label);
    if (alert >= 0 && (lastAlertMs == 0 || millis() - lastAlertMs >= ALERT_COOLDOWN_MS)) {
        announce((size_t)alert);
    }
}