/*
 * ================================================================
 * AI SMART GLASSES - ESP32-S3 + OV3660 + EDGE IMPULSE FOMO
 * ================================================================
 *
 * Model contains five classes:
 *   car
 *   chair
 *   door
 *   person
 *   stop sign
 *
 * Audio:
 *   Each class maps to one track in /01.
 *
 * Boot:
 *   Plays car audio once as YX5300 hardware test.
 *
 * Camera:
 *   OV3660 JPEG 320x240
 *   JPEG -> RGB888
 *   RGB888 -> 96x96
 *   Edge Impulse FOMO
 *
 * Serial:
 *   115200 baud
 * ================================================================
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

#define CAMERA_WIDTH       320
#define CAMERA_HEIGHT      240
#define INFERENCE_WIDTH    160
#define INFERENCE_HEIGHT   120

#define MODEL_WIDTH        EI_CLASSIFIER_INPUT_WIDTH
#define MODEL_HEIGHT       EI_CLASSIFIER_INPUT_HEIGHT

#define ENABLE_AUDIO       1

// Detection threshold.
// Edge Impulse export threshold is 0.50 in your model.
// Start with 0.50 for testing.
#define DETECTION_THRESHOLD 0.50f

// Prevent repeated detections from interrupting the current announcement.
#define AUDIO_COOLDOWN_MS   3000UL

// Keep the camera/web stream responsive while avoiding redundant inference.
// The latest frame is always copied for the web; inference runs at this rate.
#define INFERENCE_INTERVAL_MS 250UL

// Camera orientation
#define CAMERA_VFLIP        1
#define CAMERA_HMIRROR      0


// ================================================================
// MODEL CHECKS
// ================================================================

#if EI_CLASSIFIER_OBJECT_DETECTION != 1
#error "This is not an Edge Impulse object detection model."
#endif

static_assert(EI_CLASSIFIER_INPUT_WIDTH == 96,
              "Expected 96x96 Edge Impulse model.");

static_assert(EI_CLASSIFIER_INPUT_HEIGHT == 96,
              "Expected 96x96 Edge Impulse model.");


// ================================================================
// IMAGE BUFFERS
// ================================================================

constexpr size_t MODEL_PIXELS =
    (size_t)MODEL_WIDTH * MODEL_HEIGHT;

constexpr size_t MODEL_RGB_BYTES =
    MODEL_PIXELS * 3;

constexpr size_t FULL_RGB_BYTES =
    (size_t)INFERENCE_WIDTH * INFERENCE_HEIGHT * 3;


// 96x96 RGB image
static uint8_t modelRgb[MODEL_RGB_BYTES];

// 320x240 decoded RGB image
static uint8_t *fullRgb = nullptr;
static uint8_t *latestJpeg = nullptr;
static size_t latestJpegLength = 0;

static WebServer webServer(80);
static char latestDetection[32] = "none";
static float latestConfidence = 0.0f;
static uint32_t latestInferenceMs = 0;
static uint32_t latestFrameMs = 0;
static uint32_t latestBoxes = 0;
static uint16_t latestBoxX = 0;
static uint16_t latestBoxY = 0;
static uint16_t latestBoxWidth = 0;
static uint16_t latestBoxHeight = 0;
static SemaphoreHandle_t jpegMutex = nullptr;


// ================================================================
// CAMERA TIMING
// ================================================================

struct CameraTiming {
    uint32_t grabUs;
    uint32_t decodeUs;
    uint32_t resizeUs;
};


// ================================================================
// CAMERA ERROR HELPER
// ================================================================

static void checkCameraSetting(const char *name, int rc)
{
    if (rc != 0) {
        Serial.printf("[CAM] WARNING: %s returned %d\n",
                      name, rc);
    }
}


// ================================================================
// EDGE IMPULSE STYLE SQUASH RESIZE
// ================================================================
//
// 320x240 RGB888 -> 96x96 RGB888
//
// This is squash resizing, matching the image processing approach
// used by the Edge Impulse image pipeline.
// ================================================================

static bool squashRgb888(
    const uint8_t *src,
    int srcWidth,
    int srcHeight,
    uint8_t *dst,
    int dstWidth,
    int dstHeight)
{
    if (!src || !dst) {
        return false;
    }

    if (srcWidth < 2 || srcHeight < 2) {
        return false;
    }

    constexpr int FRAC_BITS = 14;
    constexpr int FRAC_VAL  = (1 << FRAC_BITS);
    constexpr int FRAC_MASK = FRAC_VAL - 1;

    constexpr int PIXEL_SIZE = 3;

    const uint32_t srcXFrac =
        ((uint32_t)srcWidth * FRAC_VAL) / dstWidth;

    const uint32_t srcYFrac =
        ((uint32_t)srcHeight * FRAC_VAL) / dstHeight;

    const int srcWidthBytes =
        srcWidth * PIXEL_SIZE;

    uint32_t srcYAccum = 0;

    for (int y = 0; y < dstHeight; y++) {

        const int ty = srcYAccum >> FRAC_BITS;

        const uint32_t yFrac =
            srcYAccum & FRAC_MASK;

        srcYAccum += srcYFrac;

        const uint32_t nyFrac =
            FRAC_VAL - yFrac;

        const uint8_t *s =
            &src[(size_t)ty * srcWidthBytes];

        uint8_t *d =
            &dst[(size_t)y * dstWidth * PIXEL_SIZE];

        uint32_t srcXAccum = 0;

        for (int x = 0; x < dstWidth; x++) {

            const int tx =
                (srcXAccum >> FRAC_BITS) * PIXEL_SIZE;

            const uint32_t xFrac =
                srcXAccum & FRAC_MASK;

            srcXAccum += srcXFrac;

            const uint32_t nxFrac =
                FRAC_VAL - xFrac;

            for (int c = 0; c < PIXEL_SIZE; c++) {

                uint32_t p00 = s[tx + c];

                uint32_t p10 = s[tx + PIXEL_SIZE + c];

                uint32_t p01 =
                    s[tx + srcWidthBytes + c];

                uint32_t p11 =
                    s[tx + srcWidthBytes + PIXEL_SIZE + c];

                p00 =
                    ((p00 * nxFrac) +
                     (p10 * xFrac) +
                     FRAC_VAL / 2) >> FRAC_BITS;

                p01 =
                    ((p01 * nxFrac) +
                     (p11 * xFrac) +
                     FRAC_VAL / 2) >> FRAC_BITS;

                p00 =
                    ((p00 * nyFrac) +
                     (p01 * yFrac) +
                     FRAC_VAL / 2) >> FRAC_BITS;

                *d++ = (uint8_t)p00;
            }
        }
    }

    return true;
}


// ================================================================
// CAMERA INITIALIZATION
// ================================================================

static bool beginCamera()
{
    Serial.println();
    Serial.println("[CAM] Initializing OV3660...");

    if (!psramFound()) {

        Serial.println(
            "[CAM] ERROR: PSRAM not detected."
        );

        return false;
    }

    camera_config_t cfg = {};

    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;

    // ------------------------------------------------------------
    // YOUR ESP32-S3 OV3660 PINS
    // ------------------------------------------------------------

    cfg.pin_d0 = 11;
    cfg.pin_d1 = 9;
    cfg.pin_d2 = 8;
    cfg.pin_d3 = 10;
    cfg.pin_d4 = 12;
    cfg.pin_d5 = 18;
    cfg.pin_d6 = 17;
    cfg.pin_d7 = 16;

    cfg.pin_xclk  = 15;
    cfg.pin_pclk  = 13;
    cfg.pin_vsync = 6;
    cfg.pin_href  = 7;

    cfg.pin_sccb_sda = 4;
    cfg.pin_sccb_scl = 5;

    cfg.pin_pwdn  = -1;
    cfg.pin_reset = -1;

    // ------------------------------------------------------------
    // CAMERA SETTINGS
    // ------------------------------------------------------------

    cfg.xclk_freq_hz = 20000000;

    cfg.pixel_format = PIXFORMAT_JPEG;

    cfg.frame_size = FRAMESIZE_QVGA;

    cfg.jpeg_quality = 10;

    cfg.fb_count = 2;

    cfg.fb_location = CAMERA_FB_IN_PSRAM;

    cfg.grab_mode = CAMERA_GRAB_LATEST;


    const esp_err_t err =
        esp_camera_init(&cfg);

    if (err != ESP_OK) {

        Serial.printf(
            "[CAM] ERROR: esp_camera_init failed: 0x%x\n",
            (unsigned)err
        );

        return false;
    }


    sensor_t *sensor =
        esp_camera_sensor_get();

    if (!sensor) {

        Serial.println(
            "[CAM] ERROR: sensor not found."
        );

        esp_camera_deinit();

        return false;
    }


    Serial.printf(
        "[CAM] Sensor PID = 0x%04X\n",
        (unsigned)sensor->id.PID
    );


    // ------------------------------------------------------------
    // IMAGE ORIENTATION
    // ------------------------------------------------------------

    checkCameraSetting(
        "vflip",
        sensor->set_vflip(
            sensor,
            CAMERA_VFLIP
        )
    );

    checkCameraSetting(
        "hmirror",
        sensor->set_hmirror(
            sensor,
            CAMERA_HMIRROR
        )
    );


    // ------------------------------------------------------------
    // IMAGE SETTINGS
    // ------------------------------------------------------------

    checkCameraSetting(
        "brightness",
        sensor->set_brightness(sensor, 1)
    );

    checkCameraSetting(
        "contrast",
        sensor->set_contrast(sensor, 0)
    );

    checkCameraSetting(
        "saturation",
        sensor->set_saturation(sensor, 0)
    );

    checkCameraSetting(
        "white balance",
        sensor->set_whitebal(sensor, 1)
    );

    checkCameraSetting(
        "AWB gain",
        sensor->set_awb_gain(sensor, 1)
    );

    checkCameraSetting(
        "WB mode",
        sensor->set_wb_mode(sensor, 0)
    );

    checkCameraSetting(
        "gain control",
        sensor->set_gain_ctrl(sensor, 1)
    );

    checkCameraSetting(
        "exposure control",
        sensor->set_exposure_ctrl(sensor, 1)
    );

    checkCameraSetting(
        "colorbar",
        sensor->set_colorbar(sensor, 0)
    );


    // ------------------------------------------------------------
    // FULL RGB BUFFER
    // ------------------------------------------------------------

    fullRgb =
        (uint8_t *)heap_caps_malloc(
            FULL_RGB_BYTES,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
        );


    if (!fullRgb) {

        Serial.printf(
            "[CAM] ERROR: Could not allocate %u bytes RGB buffer.\n",
            (unsigned)FULL_RGB_BYTES
        );

        esp_camera_deinit();

        return false;
    }

    latestJpeg =
        (uint8_t *)heap_caps_malloc(
            128 * 1024,
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
        );

    if (!latestJpeg) {
        Serial.println("[CAM] ERROR: Could not allocate web JPEG buffer.");
        heap_caps_free(fullRgb);
        fullRgb = nullptr;
        esp_camera_deinit();
        return false;
    }



    // ------------------------------------------------------------
    // CAMERA SETTLE
    // ------------------------------------------------------------

    Serial.println(
        "[CAM] Letting exposure/WB settle..."
    );

    uint32_t start = millis();

    while ((uint32_t)(millis() - start) < 1500UL) {

        camera_fb_t *fb =
            esp_camera_fb_get();

        if (fb) {
            esp_camera_fb_return(fb);
        }

        delay(20);
    }


    Serial.printf(
        "[CAM] PSRAM total=%u bytes\n",
        (unsigned)ESP.getPsramSize()
    );

    Serial.printf(
        "[CAM] PSRAM free=%u bytes\n",
        (unsigned)ESP.getFreePsram()
    );

    Serial.printf(
        "[CAM] Heap free=%u bytes\n",
        (unsigned)ESP.getFreeHeap()
    );

    Serial.println(
        "[CAM] OV3660 READY."
    );

    return true;
}


// ================================================================
// JPEG READER
// ================================================================

struct JpegContext {

    const uint8_t *src;

    size_t len;

    uint8_t *dst;

    uint16_t width;
    uint16_t height;

    bool dimensionsOk;
};


static size_t jpegRead(
    void *arg,
    size_t index,
    uint8_t *buf,
    size_t len)
{
    JpegContext *ctx =
        (JpegContext *)arg;

    if (!ctx) {
        return 0;
    }

    if (index >= ctx->len) {
        return 0;
    }

    if (len > ctx->len - index) {
        len = ctx->len - index;
    }

    if (buf) {
        memcpy(
            buf,
            ctx->src + index,
            len
        );
    }

    return len;
}


// ================================================================
// JPEG WRITER
// ================================================================

static bool jpegWrite(
    void *arg,
    uint16_t x,
    uint16_t y,
    uint16_t w,
    uint16_t h,
    uint8_t *data)
{
    JpegContext *ctx =
        (JpegContext *)arg;

    if (!ctx) {
        return false;
    }


    // First callback checks dimensions.
    if (!data) {

        if (x == 0 && y == 0) {

            ctx->dimensionsOk =
                (w == ctx->width &&
                 h == ctx->height);
        }

        return ctx->dimensionsOk;
    }


    if (!ctx->dimensionsOk) {
        return false;
    }


    if ((uint32_t)x + w > ctx->width ||
        (uint32_t)y + h > ctx->height) {

        return false;
    }


    for (uint16_t row = 0;
         row < h;
         row++) {

        memcpy(
            ctx->dst +
                ((size_t)(y + row) *
                 ctx->width +
                 x) * 3,

            data +
                (size_t)row *
                w * 3,

            (size_t)w * 3
        );
    }

    return true;
}


// ================================================================
// CAPTURE + PREPROCESS
// ================================================================

static bool captureModel(
    CameraTiming &timing,
    bool prepareModel)
{
    memset(
        &timing,
        0,
        sizeof(timing)
    );


    // ------------------------------------------------------------
    // CAMERA FRAME
    // ------------------------------------------------------------

    uint32_t t0 = micros();

    camera_fb_t *fb =
        esp_camera_fb_get();

    timing.grabUs =
        micros() - t0;


    if (!fb) {

        Serial.println(
            "[CAM] ERROR: fb_get() failed."
        );

        return false;
    }


    if (fb->format != PIXFORMAT_JPEG ||
        fb->width != CAMERA_WIDTH ||
        fb->height != CAMERA_HEIGHT) {

        Serial.printf(
            "[CAM] ERROR: format=%d size=%ux%u\n",
            (int)fb->format,
            (unsigned)fb->width,
            (unsigned)fb->height
        );

        esp_camera_fb_return(fb);

        return false;
    }


    if (fb->len <= 128 * 1024) {
        if (jpegMutex) {
            xSemaphoreTake(jpegMutex, portMAX_DELAY);
        }
        memcpy(latestJpeg, fb->buf, fb->len);
        latestJpegLength = fb->len;
        latestFrameMs = millis();
        if (jpegMutex) {
            xSemaphoreGive(jpegMutex);
        }
    }

    if (!prepareModel) {
        esp_camera_fb_return(fb);
        return true;
    }

    // ------------------------------------------------------------
    // JPEG -> RGB888 at half resolution for faster inference
    // ------------------------------------------------------------

    JpegContext ctx = {
        fb->buf,
        fb->len,
        fullRgb,
        INFERENCE_WIDTH,
        INFERENCE_HEIGHT,
        false
    };


    t0 = micros();

    esp_err_t jpegErr =
        esp_jpg_decode(
            fb->len,
            JPG_SCALE_2X,
            jpegRead,
            jpegWrite,
            &ctx
        );

    timing.decodeUs =
        micros() - t0;


    if (jpegErr != ESP_OK ||
        !ctx.dimensionsOk) {

        Serial.printf(
            "[CAM] ERROR: JPEG decode failed: 0x%x\n",
            (unsigned)jpegErr
        );

        esp_camera_fb_return(fb);

        return false;
    }

        esp_camera_fb_return(fb);


    // ------------------------------------------------------------
    // RGB888 160x120 -> RGB888 96x96
    // ------------------------------------------------------------

    t0 = micros();

    bool resizeOk =
        squashRgb888(
            fullRgb,
            INFERENCE_WIDTH,
            INFERENCE_HEIGHT,
            modelRgb,
            MODEL_WIDTH,
            MODEL_HEIGHT
        );

    timing.resizeUs =
        micros() - t0;


    if (!resizeOk) {

        Serial.println(
            "[IMAGE] ERROR: resize failed."
        );

        return false;
    }


    return true;
}

static void handleWebRoot()
{
    static const char page[] = R"HTML(
<!doctype html><html><head><meta name="viewport" content="width=device-width">
<title>AI Smart Glasses</title>
<style>
body{font-family:system-ui;background:#111;color:#eee;margin:16px}
img{width:min(100%,640px);image-rendering:auto;border:2px solid #555}
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
    if (!latestJpeg || latestJpegLength == 0) {
        webServer.send(503, "text/plain", "No camera frame available");
        return;
    }

    webServer.sendHeader("Cache-Control", "no-store");
    webServer.sendHeader("Connection", "close");
    if (jpegMutex) {
        xSemaphoreTake(jpegMutex, portMAX_DELAY);
    }
    const size_t frameLength = latestJpegLength;
    webServer.setContentLength(frameLength);
    webServer.send(200, "image/jpeg", "");
    WiFiClient client = webServer.client();
    client.write(latestJpeg, frameLength);
    if (jpegMutex) {
        xSemaphoreGive(jpegMutex);
    }
}

static void handleWebStatus()
{
    char json[256];
    snprintf(
        json,
        sizeof(json),
        "{\"detection\":\"%s\",\"confidence\":%.3f,\"boxes\":%lu,\"x\":%u,\"y\":%u,\"width\":%u,\"height\":%u,\"inference_ms\":%lu,\"frame_age_ms\":%lu}",
        latestDetection,
        (double)latestConfidence,
        (unsigned long)latestBoxes,
        (unsigned)latestBoxX,
        (unsigned)latestBoxY,
        (unsigned)latestBoxWidth,
        (unsigned)latestBoxHeight,
        (unsigned long)latestInferenceMs,
        (unsigned long)(millis() - latestFrameMs)
    );
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
    jpegMutex = xSemaphoreCreateMutex();
    if (!jpegMutex) {
        Serial.println("[WEB] ERROR: JPEG mutex allocation failed.");
        return;
    }
    xTaskCreatePinnedToCore(
        webServerTask,
        "web-server",
        4096,
        nullptr,
        1,
        nullptr,
        0
    );

    Serial.printf("[WEB] Connect to Wi-Fi '%s' password '%s'\n", ssid, password);
    Serial.printf("[WEB] Open http://%s/\n", WiFi.softAPIP().toString().c_str());
}


// ================================================================
// EDGE IMPULSE IMAGE SIGNAL
// ================================================================
//
// Edge Impulse expects one float per pixel.
// For RGB images, the pixel is packed as:
//
//     0xRRGGBB
//
// Example:
//     R=255 G=0 B=0
//
//     0xFF0000
//
// This is NOT 3 floats per pixel.
// ================================================================

static int getData(
    size_t offset,
    size_t length,
    float *out)
{
    if (!out) {
        return -1;
    }

    if (offset > MODEL_PIXELS) {
        return -1;
    }

    if (length >
        MODEL_PIXELS - offset) {

        return -1;
    }


    const uint8_t *p =
        modelRgb + offset * 3;


    for (size_t i = 0;
         i < length;
         i++) {

        uint32_t rgb =
            ((uint32_t)p[0] << 16) |
            ((uint32_t)p[1] << 8) |
            ((uint32_t)p[2]);

        out[i] =
            (float)rgb;

        p += 3;
    }


    return EIDSP_OK;
}


// ================================================================
// AUDIO
// ================================================================

#if ENABLE_AUDIO

constexpr int AUDIO_RX_PIN = 2;
constexpr int AUDIO_TX_PIN = 1;

constexpr uint8_t AUDIO_VOLUME = 25;

constexpr uint8_t AUDIO_FOLDER = 1;


// ---------------------------------------------------------------
// Keep the track order aligned with the five model labels and /01/001.mp3...
// ---------------------------------------------------------------

struct AudioMapping {

    const char *label;

    uint8_t track;

    uint32_t lastPlayed;
};


static AudioMapping audioMap[] = {

    {
        "car",
        1,
        0
    },
    {
        "chair",
        2,
        0
    },
    {
        "door",
        3,
        0
    },
    {
        "person",
        4,
        0
    },
    {
        "stop sign",
        5,
        0
    }

};


constexpr size_t AUDIO_COUNT =
    sizeof(audioMap) /
    sizeof(audioMap[0]);


static YX5300_ESP32 *player = nullptr;
static uint32_t lastAudioPlayed = 0;


// ---------------------------------------------------------------
// Find audio mapping
// ---------------------------------------------------------------

static int findAudio(
    const char *label)
{
    if (!label) {
        return -1;
    }


    for (size_t i = 0;
         i < AUDIO_COUNT;
         i++) {

        if (strcmp(
                label,
                audioMap[i].label) == 0) {

            return (int)i;
        }
    }


    return -1;
}


// ---------------------------------------------------------------
// Play audio
// ---------------------------------------------------------------

static void playAudio(
    size_t index)
{
    if (!player) {
        Serial.println(
            "[AUDIO] ERROR: player is NULL."
        );

        return;
    }


    if (index >= AUDIO_COUNT) {
        return;
    }


    const char *label =
        audioMap[index].label;

    uint8_t track =
        audioMap[index].track;


    Serial.printf(
        "[AUDIO] PLAY: %s | folder=%02u | track=%03u\n",
        label,
        (unsigned)AUDIO_FOLDER,
        (unsigned)track
    );


    player->playTrackInFolder(
        track,
        AUDIO_FOLDER
    );


    audioMap[index].lastPlayed = millis();
    lastAudioPlayed = audioMap[index].lastPlayed;
}


// ---------------------------------------------------------------
// Start YX5300
// ---------------------------------------------------------------

static void beginAudio()
{
    Serial.println();
    Serial.println(
        "[AUDIO] Initializing YX5300..."
    );


    delay(1000);


    // Keep object alive for entire program.
    static YX5300_ESP32 module(
        Serial2,
        AUDIO_RX_PIN,
        AUDIO_TX_PIN
    );


    player = &module;


    delay(500);


    player->setVolume(
        AUDIO_VOLUME
    );

    player->unmute();

    delay(300);


    Serial.println(
        "[AUDIO] YX5300 initialized."
    );

    Serial.printf(
        "[AUDIO] UART RX=GPIO%d TX=GPIO%d\n",
        AUDIO_RX_PIN,
        AUDIO_TX_PIN
    );

    Serial.printf(
        "[AUDIO] Volume=%u\n",
        (unsigned)AUDIO_VOLUME
    );

    Serial.printf(
        "[AUDIO] Tracks on TF card=%d | module volume=%d\n",
        player->queryTrackCount(),
        player->queryCurrentVolume()
    );


    // ------------------------------------------------------------
    // BOOT AUDIO TEST
    // ------------------------------------------------------------

    Serial.println(
        "[AUDIO] Playing CAR test audio..."
    );

    playAudio(0);


    // Do not block the web/inference loop while the module starts playback.
}

#endif


// ================================================================
// PRINT MODEL INFORMATION
// ================================================================

static void printModelInfo()
{
    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "EDGE IMPULSE MODEL INFORMATION"
    );

    Serial.println(
        "========================================"
    );


    Serial.printf(
        "Input: %dx%d\n",
        EI_CLASSIFIER_INPUT_WIDTH,
        EI_CLASSIFIER_INPUT_HEIGHT
    );


    Serial.printf(
        "Labels: %d\n",
        EI_CLASSIFIER_LABEL_COUNT
    );


    Serial.printf(
        "Object detection: %d\n",
        EI_CLASSIFIER_OBJECT_DETECTION
    );


    Serial.printf(
        "Detection threshold: %.2f\n",
        (double)
        EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD
    );


    Serial.println(
        "Model labels:"
    );


    for (int i = 0;
         i < EI_CLASSIFIER_LABEL_COUNT;
         i++) {

        Serial.printf(
            "  [%d] %s\n",
            i,
            ei_classifier_inferencing_categories[i]
        );
    }


    Serial.println();
    Serial.println("AUDIO CLASSES (/01): car=001 chair=002 door=003 person=004 stop sign=005");


    Serial.println(
        "========================================"
    );

    Serial.println();
}


// ================================================================
// SETUP
// ================================================================

void setup()
{
    Serial.begin(115200);

    delay(2000);


    Serial.println();
    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "AI SMART GLASSES"
    );

    Serial.println(
        "ESP32-S3 + OV3660 + Edge Impulse FOMO"
    );

    Serial.println("FIVE-CLASS REAL-TIME DETECTION");

    Serial.println(
        "========================================"
    );


    // ------------------------------------------------------------
    // Model information
    // ------------------------------------------------------------

    printModelInfo();


    // ------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------

    if (!beginCamera()) {

        Serial.println(
            "[BOOT] CAMERA FAILED."
        );

        while (true) {
            delay(1000);
        }
    }


    // ------------------------------------------------------------
    // Edge Impulse
    // ------------------------------------------------------------

    Serial.println(
        "[EI] Initializing classifier..."
    );


    run_classifier_init();


    Serial.println(
        "[EI] Classifier initialized."
    );

    beginWebServer();


    // ------------------------------------------------------------
    // Audio
    // ------------------------------------------------------------

#if ENABLE_AUDIO

    beginAudio();

#endif


    Serial.println();
    Serial.println(
        "[BOOT] SYSTEM READY."
    );

    Serial.println("[BOOT] Detecting car, chair, door, person, and stop sign.");

    Serial.println(
        "[BOOT] Serial monitor = 115200."
    );

    Serial.println();
}


// ================================================================
// MAIN LOOP
// ================================================================

void loop()
{
    static uint32_t lastTimingLog = 0;
    static uint32_t lastInferenceStart = 0;

    CameraTiming timing;
    const bool runInference =
        (lastInferenceStart == 0) ||
        (millis() - lastInferenceStart >= INFERENCE_INTERVAL_MS);


    // ------------------------------------------------------------
    // CAPTURE
    // ------------------------------------------------------------

    if (!captureModel(timing, runInference)) {

        Serial.println(
            "[LOOP] Capture failed."
        );

        delay(100);

        return;
    }

    if (!runInference) {
        delay(1);
        return;
    }

    // ------------------------------------------------------------
    // EDGE IMPULSE SIGNAL
    // ------------------------------------------------------------

    signal_t signal = {};

    signal.total_length =
        MODEL_PIXELS;

    signal.get_data =
        getData;


    // ------------------------------------------------------------
    // RUN MODEL
    // ------------------------------------------------------------

    ei_impulse_result_t result = {};


    uint32_t inferenceStart =
        millis();
    lastInferenceStart = inferenceStart;


    EI_IMPULSE_ERROR err =
        run_classifier(
            &signal,
            &result,
            false
        );


    uint32_t inferenceTime =
        millis() - inferenceStart;
    // Schedule the next inference after this one finishes. Measuring from
    // the start caused a long inference to immediately run again.
    lastInferenceStart = millis();


    if (err != EI_IMPULSE_OK) {

        Serial.printf(
            "[EI] ERROR: run_classifier() = %d\n",
            (int)err
        );

        delay(200);

        return;
    }

    // ------------------------------------------------------------
    // PRINT TIMING
    // ------------------------------------------------------------

    if (millis() - lastTimingLog >= 1000UL) {
        Serial.printf(
            "[TIME] grab=%.1f ms | decode=%.1f ms | resize=%.1f ms | DSP=%lu ms | NN=%lu ms | total=%lu ms\n",
            timing.grabUs / 1000.0,
            timing.decodeUs / 1000.0,
            timing.resizeUs / 1000.0,
            (unsigned long)result.timing.dsp,
            (unsigned long)result.timing.classification,
            (unsigned long)inferenceTime
        );
        lastTimingLog = millis();
    }


    // ------------------------------------------------------------
    // DETECTIONS
    // ------------------------------------------------------------

    if (result.bounding_boxes_count == 0) {
        strcpy(latestDetection, "none");
        latestConfidence = 0.0f;
        latestBoxes = 0;
        latestBoxX = 0;
        latestBoxY = 0;
        latestBoxWidth = 0;
        latestBoxHeight = 0;
        latestInferenceMs = inferenceTime;

        Serial.println(
            "[DETECT] No bounding boxes."
        );

        return;
    }

    float bestConfidence = 0.0f;
    const char *bestLabel = nullptr;
    uint32_t bestBoxIndex = 0;

    for (uint32_t i = 0;
         i < result.bounding_boxes_count;
         i++) {
        const auto &bb = result.bounding_boxes[i];
        if (bb.label && bb.value > bestConfidence) {
            bestConfidence = bb.value;
            bestLabel = bb.label;
            bestBoxIndex = i;
        }
    }

    strncpy(
        latestDetection,
        bestLabel ? bestLabel : "none",
        sizeof(latestDetection) - 1
    );
    latestDetection[sizeof(latestDetection) - 1] = '\0';
    latestConfidence = bestConfidence;
    latestBoxes = result.bounding_boxes_count;
    latestBoxX = result.bounding_boxes[bestBoxIndex].x;
    latestBoxY = result.bounding_boxes[bestBoxIndex].y;
    latestBoxWidth = result.bounding_boxes[bestBoxIndex].width;
    latestBoxHeight = result.bounding_boxes[bestBoxIndex].height;
    latestInferenceMs = inferenceTime;

#if ENABLE_AUDIO
    // Only announce the strongest object. Sending one command per bounding
    // box makes the YX5300 interrupt itself and can make playback appear stuck.
    if (bestLabel && bestConfidence >= DETECTION_THRESHOLD) {
        const int audioIndex = findAudio(bestLabel);
        const uint32_t now = millis();
        const bool classReady =
            audioIndex >= 0 &&
            (audioMap[audioIndex].lastPlayed == 0 ||
             now - audioMap[audioIndex].lastPlayed >= AUDIO_COOLDOWN_MS);
        const bool playerReady =
            lastAudioPlayed == 0 ||
            now - lastAudioPlayed >= AUDIO_COOLDOWN_MS;

        if (classReady && playerReady) {
            Serial.printf("[AUDIO] %s %.3f\n", bestLabel, (double)bestConfidence);
            playAudio((size_t)audioIndex);
        }
    }
#endif

}