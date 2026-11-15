#include "main.h"
#include "adc.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

#include <stdio.h>
#include "OLED12864.h"

#define KEY_S1_PORT GPIOB
#define KEY_S1_PIN  GPIO_PIN_0

#define KEY_S2_PORT GPIOB
#define KEY_S2_PIN  GPIO_PIN_1

#define KEY_S3_PORT GPIOB
#define KEY_S3_PIN  GPIO_PIN_10

#define KEY_S4_PORT GPIOB
#define KEY_S4_PIN  GPIO_PIN_11

#define CHG_PWM_TIM      htim3
#define CHG_PWM_CHANNEL  TIM_CHANNEL_1

#define LED_PWM_TIM      htim3
#define LED_PWM_CHANNEL  TIM_CHANNEL_2

#define ADC_CH_VBAT      ADC_CHANNEL_1
#define ADC_CH_VSL       ADC_CHANNEL_2
#define ADC_CH_ILED      ADC_CHANNEL_3
#define ADC_CH_ICHARGE   ADC_CHANNEL_4

#define ADC_VREF         3.30f
#define ADC_MAX_VALUE    4095.0f

#define VBAT_DIV_RATIO   3.1277f
#define VSL_DIV_RATIO    3.1277f

#define CHG_SAMPLE_R     0.33f
#define CHG_AMP_GAIN     10.0f

#define GOAL_V           8.20f
#define RESTART_V        7.80f

#define TAPER_START_V    8.10f
#define HARD_STOP_V      8.25f

#define I_STOP_MA        100.0f

#define I_FULL_MA        200.0f
#define I_AT_GOAL_MA     135.0f
#define I_AT_HARD_MA     100.0f

#define TAPER_PWM_AT_GOAL 75
#define TAPER_PWM_AT_HARD 55

#define FINISH_INIT_PWM  100

#define CHG_CONTROL_PERIOD_MS 500
#define I_STOP_CONFIRM_COUNT 2

#define I_TAPER_HIGH_TOL_MA 12.0f
#define I_TAPER_LOW_TOL_MA  25.0f

#define LED_SAMPLE_R     0.33f
#define LED_AMP_GAIN     10.0f

#define LED_TARGET_DEFAULT_MA 300.0f
#define LED_TARGET_MIN_MA     50.0f
#define LED_TARGET_MAX_MA     300.0f
#define LED_TARGET_STEP_MA    25.0f

#define LED_TOL_MA       15.0f
#define LED_SAFE_MAX_MA  340.0f

#define LED_DUTY_MAX     90
#define LED_DUTY_MIN     0

#define FLOW_DUTY_MIN    25
#define FLOW_DUTY_MAX    90
#define FLOW_DUTY_STEP   10
#define FLOW_PERIOD_MS   300

#define VBAT_LED_OFF_V   6.00f
#define VBAT_LED_ON_V    6.40f

#define LED_CONTROL_PERIOD_MS 250

#define DISPLAY_STATE_SPLASH 6
#define DISPLAY_STATE_NORMAL 7

#define PWM_INVERTED 1

#define KEY_NONE        0
#define KEY_SHORT       1
#define KEY_LONG        2
#define KEY_LONG_MS     800

volatile uint8_t display_state = DISPLAY_STATE_NORMAL;

typedef enum
{
    STATE_STOP = 0,
    STATE_NORMAL_CHARGE = 1,
    STATE_FINISH = 2
} ChargeState;

typedef enum
{
    CHG_MODE_AUTO = 0,
    CHG_MODE_MANUAL = 1
} ChargeWorkMode;

typedef enum
{
    MODE_STOP = 0,
    MODE_AUTO = 1,
    MODE_FLOW = 2
} LedWorkMode;

typedef enum
{
    PAGE_DAY_CHARGE = 0,
    PAGE_NIGHT_LED = 1
} DisplayPage;

ChargeWorkMode charge_work_mode = CHG_MODE_AUTO;
ChargeState charge_state = STATE_STOP;

LedWorkMode led_work_mode = MODE_STOP;
DisplayPage display_page = PAGE_DAY_CHARGE;

uint16_t raw_vbat = 0;
uint16_t raw_vsl = 0;
uint16_t raw_icharge = 0;
uint16_t raw_iled = 0;

float adc_vbat_voltage = 0.0f;
float adc_vsl_voltage = 0.0f;
float adc_icharge_voltage = 0.0f;
float adc_iled_voltage = 0.0f;

float vbat_voltage = 0.0f;
float vsl_voltage = 0.0f;

float icharge_sample_mV = 0.0f;
float icharge_current_mA = 0.0f;
float icharge_filter_mA = 0.0f;

float target_current_mA = I_FULL_MA;

uint8_t pwm_voltage = 0;
uint8_t bypass_duty = 100;

uint32_t last_control_tick = 0;

uint8_t stop_current_low_count = 0;

uint8_t led_duty = 0;
uint8_t pb5_duty = 100;
uint8_t low_bat_lock = 0;

uint8_t flow_duty = FLOW_DUTY_MIN;
uint8_t flow_dir = 1;

float led_sample_mV = 0.0f;
float led_current_mA = 0.0f;
float led_current_filter_mA = 0.0f;

float led_target_mA = LED_TARGET_DEFAULT_MA;

uint32_t last_led_control_tick = 0;
uint32_t last_flow_tick = 0;

void SystemClock_Config(void);

static void Key_GPIO_Init_User(void);
static uint8_t Key_IsPressed(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);
static uint8_t Key_ScanEx(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);

static uint16_t ADC_Read_Channel(uint32_t channel);
static uint16_t ADC_Read_Channel_Average(uint32_t channel, uint8_t times);

static void Charge_SetPWM(uint8_t pwm);
static void Charge_PWM_Inc(uint8_t step);
static void Charge_PWM_Dec(uint8_t step);

static void Charge_UpdateMeasure(void);
static void Charge_AutoControl(void);
static void Charge_FinishControl(void);

static void Charge_EnterStop(void);
static void Charge_EnterNormal(void);
static void Charge_EnterFinish(void);

static void LED_SetDuty(uint8_t duty);
static void LED_UpdateMeasure(void);
static void LED_AutoControl(void);
static void LED_FlowControl(void);
static void LED_BatteryProtect(void);

static void LED_EnterAuto(void);
static void LED_EnterStop(void);
static void LED_EnterFlow(void);

static void OLED_TestScreen(void);
static void OLED_ShowInfo(void);
static void OLED_ShowChargePage(void);
static void OLED_ShowLEDPage(void);
static void UART_PrintInfo(void);

int main(void)
{
    uint32_t last_uart_time = 0;

    uint8_t key1;
    uint8_t key2;
    uint8_t key3;
    uint8_t key4;

    HAL_Init();

    SystemClock_Config();

    MX_GPIO_Init();
    MX_ADC1_Init();
    MX_I2C1_Init();
    MX_TIM3_Init();
    MX_USART1_UART_Init();

    Key_GPIO_Init_User();

    HAL_ADCEx_Calibration_Start(&hadc1);

    HAL_TIM_PWM_Start(&CHG_PWM_TIM, CHG_PWM_CHANNEL);
    HAL_TIM_PWM_Start(&LED_PWM_TIM, LED_PWM_CHANNEL);

    Charge_EnterStop();
    LED_EnterStop();

    HAL_Delay(500);

    OLED_Init();
    HAL_Delay(100);
    OLED_Clear();
    HAL_Delay(100);

    OLED_TestScreen();
    HAL_Delay(1500);
    OLED_Clear();

    Charge_UpdateMeasure();
    LED_UpdateMeasure();

    icharge_filter_mA = icharge_current_mA;
    led_current_filter_mA = led_current_mA;

    charge_work_mode = CHG_MODE_AUTO;

    if (vbat_voltage < RESTART_V)
    {
        Charge_EnterNormal();
    }
    else
    {
        Charge_EnterStop();
    }

    printf("\r\nSolar Lamp Merge Start\r\n");
    printf("S4 Long: DAY/NIGHT display\r\n");

    while (1)
    {
        key1 = KEY_NONE;
        key2 = KEY_NONE;
        key3 = KEY_NONE;
        key4 = KEY_NONE;

        Charge_UpdateMeasure();

        if (charge_work_mode == CHG_MODE_AUTO)
        {
            Charge_AutoControl();
        }

        LED_UpdateMeasure();
        LED_BatteryProtect();

        key1 = Key_ScanEx(KEY_S1_PORT, KEY_S1_PIN);
        key2 = Key_ScanEx(KEY_S2_PORT, KEY_S2_PIN);
        key3 = Key_ScanEx(KEY_S3_PORT, KEY_S3_PIN);
        key4 = Key_ScanEx(KEY_S4_PORT, KEY_S4_PIN);

        if (key4 == KEY_LONG)
        {
            if (display_page == PAGE_DAY_CHARGE)
            {
                display_page = PAGE_NIGHT_LED;
            }
            else
            {
                display_page = PAGE_DAY_CHARGE;
            }

            OLED_Clear();
        }
        else
        {
            if (display_page == PAGE_DAY_CHARGE)
            {
                if (key1 == KEY_SHORT)
                {
                    charge_work_mode = CHG_MODE_AUTO;

                    if (vbat_voltage >= TAPER_START_V)
                    {
                        Charge_EnterFinish();
                    }
                    else
                    {
                        Charge_EnterNormal();
                    }
                }

                if (key2 == KEY_SHORT)
                {
                    charge_work_mode = CHG_MODE_MANUAL;
                    Charge_EnterStop();
                }

                if (key3 == KEY_SHORT)
                {
                    charge_work_mode = CHG_MODE_MANUAL;
                    Charge_PWM_Inc(5);
                }

                if (key4 == KEY_SHORT)
                {
                    charge_work_mode = CHG_MODE_MANUAL;
                    Charge_PWM_Dec(5);
                }
            }
            else
            {
                if (key1 == KEY_SHORT)
                {
                    if (low_bat_lock == 0)
                    {
                        LED_EnterAuto();
                    }
                }

                if (key2 == KEY_SHORT)
                {
                    LED_EnterStop();
                }

                if (key3 == KEY_SHORT)
                {
                    if (low_bat_lock == 0)
                    {
                        if (led_work_mode == MODE_FLOW)
                        {
                            LED_EnterAuto();
                        }
                        else
                        {
                            LED_EnterFlow();
                        }
                    }
                }

                if (key4 == KEY_SHORT)
                {
                    if (low_bat_lock == 0)
                    {
                        if (led_target_mA + LED_TARGET_STEP_MA > LED_TARGET_MAX_MA)
                        {
                            led_target_mA = LED_TARGET_MIN_MA;
                        }
                        else
                        {
                            led_target_mA += LED_TARGET_STEP_MA;
                        }

                        LED_EnterAuto();
                    }
                }
            }
        }

        if (low_bat_lock)
        {
            LED_SetDuty(0);
        }
        else
        {
            if (led_work_mode == MODE_AUTO)
            {
                LED_AutoControl();
            }
            else if (led_work_mode == MODE_FLOW)
            {
                LED_FlowControl();
            }
        }

        OLED_ShowInfo();

        if (HAL_GetTick() - last_uart_time >= 800)
        {
            last_uart_time = HAL_GetTick();
            UART_PrintInfo();
        }

        HAL_Delay(100);
    }
}

static void Charge_EnterStop(void)
{
    charge_state = STATE_STOP;

    stop_current_low_count = 0;
    target_current_mA = 0.0f;

    Charge_SetPWM(0);
}

static void Charge_EnterNormal(void)
{
    charge_state = STATE_NORMAL_CHARGE;

    stop_current_low_count = 0;
    target_current_mA = I_FULL_MA;

    Charge_SetPWM(100);
}

static void Charge_EnterFinish(void)
{
    charge_state = STATE_FINISH;

    last_control_tick = HAL_GetTick();

    stop_current_low_count = 0;

    Charge_SetPWM(FINISH_INIT_PWM);

    icharge_filter_mA = icharge_current_mA;
}

static void Charge_AutoControl(void)
{
    if (charge_state == STATE_STOP)
    {
        Charge_SetPWM(0);

        if (vbat_voltage < RESTART_V)
        {
            Charge_EnterNormal();
        }

        return;
    }

    if (charge_state == STATE_NORMAL_CHARGE)
    {
        target_current_mA = I_FULL_MA;
        Charge_SetPWM(100);

        if (vbat_voltage >= TAPER_START_V)
        {
            Charge_EnterFinish();
        }

        return;
    }

    if (charge_state == STATE_FINISH)
    {
        Charge_FinishControl();
        return;
    }
}

static void Charge_FinishControl(void)
{
    uint32_t now_tick;
    uint8_t pwm_limit;
    float ratio;

    now_tick = HAL_GetTick();

    if (vbat_voltage >= HARD_STOP_V)
    {
        Charge_EnterStop();
        return;
    }

    if ((vbat_voltage >= GOAL_V) && (icharge_filter_mA < I_STOP_MA))
    {
        if (stop_current_low_count < I_STOP_CONFIRM_COUNT)
        {
            stop_current_low_count++;
        }
    }
    else
    {
        stop_current_low_count = 0;
    }

    if (stop_current_low_count >= I_STOP_CONFIRM_COUNT)
    {
        Charge_EnterStop();
        return;
    }

    if (vbat_voltage <= TAPER_START_V)
    {
        pwm_limit = 100;
        target_current_mA = I_FULL_MA;
    }
    else if (vbat_voltage < GOAL_V)
    {
        ratio = (vbat_voltage - TAPER_START_V) / (GOAL_V - TAPER_START_V);

        if (ratio < 0.0f)
        {
            ratio = 0.0f;
        }

        if (ratio > 1.0f)
        {
            ratio = 1.0f;
        }

        pwm_limit = (uint8_t)(100.0f - ratio * (100.0f - TAPER_PWM_AT_GOAL) + 0.5f);
        target_current_mA = I_FULL_MA - ratio * (I_FULL_MA - I_AT_GOAL_MA);
    }
    else
    {
        ratio = (vbat_voltage - GOAL_V) / (HARD_STOP_V - GOAL_V);

        if (ratio < 0.0f)
        {
            ratio = 0.0f;
        }

        if (ratio > 1.0f)
        {
            ratio = 1.0f;
        }

        pwm_limit = (uint8_t)(TAPER_PWM_AT_GOAL - ratio * (TAPER_PWM_AT_GOAL - TAPER_PWM_AT_HARD) + 0.5f);
        target_current_mA = I_AT_GOAL_MA - ratio * (I_AT_GOAL_MA - I_AT_HARD_MA);
    }

    if (pwm_voltage > pwm_limit)
    {
        Charge_SetPWM(pwm_limit);
    }

    if (now_tick - last_control_tick < CHG_CONTROL_PERIOD_MS)
    {
        return;
    }

    last_control_tick = now_tick;

    if (icharge_filter_mA > target_current_mA + I_TAPER_HIGH_TOL_MA)
    {
        if (pwm_voltage > 0)
        {
            Charge_PWM_Dec(1);
        }
    }
    else if ((icharge_filter_mA < target_current_mA - I_TAPER_LOW_TOL_MA) &&
             (vbat_voltage < GOAL_V))
    {
        if (pwm_voltage < pwm_limit)
        {
            Charge_PWM_Inc(1);
        }
    }
}

static void Charge_PWM_Inc(uint8_t step)
{
    if (pwm_voltage + step >= 100)
    {
        pwm_voltage = 100;
    }
    else
    {
        pwm_voltage += step;
    }

    Charge_SetPWM(pwm_voltage);
}

static void Charge_PWM_Dec(uint8_t step)
{
    if (pwm_voltage <= step)
    {
        pwm_voltage = 0;
    }
    else
    {
        pwm_voltage -= step;
    }

    Charge_SetPWM(pwm_voltage);
}

static void Charge_SetPWM(uint8_t pwm)
{
    uint32_t period;
    uint32_t compare;

    if (pwm > 100)
    {
        pwm = 100;
    }

    pwm_voltage = pwm;
    bypass_duty = 100 - pwm_voltage;

    period = __HAL_TIM_GET_AUTORELOAD(&CHG_PWM_TIM) + 1;

    compare = period * pwm_voltage / 100;

    if (compare > period)
    {
        compare = period;
    }

    __HAL_TIM_SET_COMPARE(&CHG_PWM_TIM, CHG_PWM_CHANNEL, compare);
}

static void Charge_UpdateMeasure(void)
{
    raw_vbat = ADC_Read_Channel_Average(ADC_CH_VBAT, 64);
    raw_vsl = ADC_Read_Channel_Average(ADC_CH_VSL, 64);
    raw_icharge = ADC_Read_Channel_Average(ADC_CH_ICHARGE, 64);

    adc_vbat_voltage = raw_vbat * ADC_VREF / ADC_MAX_VALUE;
    adc_vsl_voltage = raw_vsl * ADC_VREF / ADC_MAX_VALUE;
    adc_icharge_voltage = raw_icharge * ADC_VREF / ADC_MAX_VALUE;

    vbat_voltage = adc_vbat_voltage * VBAT_DIV_RATIO;
    vsl_voltage = adc_vsl_voltage * VSL_DIV_RATIO;

    icharge_sample_mV = adc_icharge_voltage * 1000.0f / CHG_AMP_GAIN;
    icharge_current_mA = icharge_sample_mV / CHG_SAMPLE_R;

    icharge_filter_mA = icharge_filter_mA * 0.93f + icharge_current_mA * 0.07f;
}

static void LED_EnterAuto(void)
{
    led_work_mode = MODE_AUTO;

    if (led_duty < 30)
    {
        LED_SetDuty(30);
    }

    last_led_control_tick = HAL_GetTick();
}

static void LED_EnterStop(void)
{
    led_work_mode = MODE_STOP;
    LED_SetDuty(0);
}

static void LED_EnterFlow(void)
{
    led_work_mode = MODE_FLOW;

    flow_duty = FLOW_DUTY_MIN;
    flow_dir = 1;

    LED_SetDuty(flow_duty);

    last_flow_tick = HAL_GetTick();
}

static void LED_BatteryProtect(void)
{
    if (low_bat_lock == 0)
    {
        if (vbat_voltage < VBAT_LED_OFF_V)
        {
            low_bat_lock = 1;
            LED_SetDuty(0);
        }
    }
    else
    {
        if (vbat_voltage > VBAT_LED_ON_V)
        {
            low_bat_lock = 0;

            if (led_work_mode == MODE_AUTO)
            {
                if (led_duty < 30)
                {
                    LED_SetDuty(30);
                }

                last_led_control_tick = HAL_GetTick();
            }
            else if (led_work_mode == MODE_FLOW)
            {
                flow_duty = FLOW_DUTY_MIN;
                flow_dir = 1;
                LED_SetDuty(flow_duty);
                last_flow_tick = HAL_GetTick();
            }
        }
    }
}

static void LED_AutoControl(void)
{
    float err;
    uint32_t now_tick;

    now_tick = HAL_GetTick();

    if (now_tick - last_led_control_tick < LED_CONTROL_PERIOD_MS)
    {
        return;
    }

    last_led_control_tick = now_tick;

    if (led_current_filter_mA > LED_SAFE_MAX_MA)
    {
        if (led_duty >= 5)
        {
            LED_SetDuty(led_duty - 5);
        }
        else
        {
            LED_SetDuty(0);
        }

        return;
    }

    err = led_target_mA - led_current_filter_mA;

    if (err > 60.0f)
    {
        if (led_duty + 3 <= LED_DUTY_MAX)
        {
            LED_SetDuty(led_duty + 3);
        }
        else
        {
            LED_SetDuty(LED_DUTY_MAX);
        }
    }
    else if (err > 30.0f)
    {
        if (led_duty + 2 <= LED_DUTY_MAX)
        {
            LED_SetDuty(led_duty + 2);
        }
        else
        {
            LED_SetDuty(LED_DUTY_MAX);
        }
    }
    else if (err > LED_TOL_MA)
    {
        if (led_duty < LED_DUTY_MAX)
        {
            LED_SetDuty(led_duty + 1);
        }
    }
    else if (err < -60.0f)
    {
        if (led_duty >= 3)
        {
            LED_SetDuty(led_duty - 3);
        }
        else
        {
            LED_SetDuty(0);
        }
    }
    else if (err < -30.0f)
    {
        if (led_duty >= 2)
        {
            LED_SetDuty(led_duty - 2);
        }
        else
        {
            LED_SetDuty(0);
        }
    }
    else if (err < -LED_TOL_MA)
    {
        if (led_duty > 0)
        {
            LED_SetDuty(led_duty - 1);
        }
    }
}

static void LED_FlowControl(void)
{
    uint32_t now_tick;

    now_tick = HAL_GetTick();

    if (now_tick - last_flow_tick < FLOW_PERIOD_MS)
    {
        return;
    }

    last_flow_tick = now_tick;

    if (flow_dir)
    {
        if (flow_duty + FLOW_DUTY_STEP >= FLOW_DUTY_MAX)
        {
            flow_duty = FLOW_DUTY_MAX;
            flow_dir = 0;
        }
        else
        {
            flow_duty += FLOW_DUTY_STEP;
        }
    }
    else
    {
        if (flow_duty <= FLOW_DUTY_MIN + FLOW_DUTY_STEP)
        {
            flow_duty = FLOW_DUTY_MIN;
            flow_dir = 1;
        }
        else
        {
            flow_duty -= FLOW_DUTY_STEP;
        }
    }

    LED_SetDuty(flow_duty);
}

static void LED_SetDuty(uint8_t duty)
{
    uint32_t period;
    uint32_t compare;

    if (duty > LED_DUTY_MAX)
    {
        duty = LED_DUTY_MAX;
    }

    led_duty = duty;

#if PWM_INVERTED
    pb5_duty = 100 - led_duty;
#else
    pb5_duty = led_duty;
#endif

    period = __HAL_TIM_GET_AUTORELOAD(&LED_PWM_TIM) + 1;

    compare = period * pb5_duty / 100;

    if (compare > period)
    {
        compare = period;
    }

    __HAL_TIM_SET_COMPARE(&LED_PWM_TIM, LED_PWM_CHANNEL, compare);
}

static void LED_UpdateMeasure(void)
{
    raw_vbat = ADC_Read_Channel_Average(ADC_CH_VBAT, 16);
    raw_iled = ADC_Read_Channel_Average(ADC_CH_ILED, 32);

    adc_vbat_voltage = raw_vbat * ADC_VREF / ADC_MAX_VALUE;
    adc_iled_voltage = raw_iled * ADC_VREF / ADC_MAX_VALUE;

    vbat_voltage = adc_vbat_voltage * VBAT_DIV_RATIO;

    led_sample_mV = adc_iled_voltage * 1000.0f / LED_AMP_GAIN;
    led_current_mA = led_sample_mV / LED_SAMPLE_R;

    led_current_filter_mA = led_current_filter_mA * 0.80f + led_current_mA * 0.20f;
}

static void OLED_TestScreen(void)
{
    OLED_Clear();
    OLED_P6x8Str(0, 0, (unsigned char *)"Solar Lamp System");
    OLED_P6x8Str(0, 2, (unsigned char *)"S4 Long DAY/NIGHT");
    OLED_P6x8Str(0, 4, (unsigned char *)"DAY: Charge");
    OLED_P6x8Str(0, 6, (unsigned char *)"NIGHT: LED");
}

static void OLED_ShowInfo(void)
{
    if (display_page == PAGE_DAY_CHARGE)
    {
        OLED_ShowChargePage();
    }
    else
    {
        OLED_ShowLEDPage();
    }
}

static void OLED_ShowChargePage(void)
{
    char buf[32];

    int bat_x100;
    int i_show;

    bat_x100 = (int)(vbat_voltage * 100.0f + 0.5f);

    if (charge_state == STATE_FINISH)
    {
        i_show = (int)(icharge_filter_mA + 0.5f);

        sprintf(buf, "DAY FIN B:%2d.%02d",
                bat_x100 / 100,
                bat_x100 % 100);
        OLED_P6x8Str(0, 0, (unsigned char *)buf);

        sprintf(buf, "I:%4dmA        ",
                i_show);
        OLED_P6x8Str(0, 2, (unsigned char *)buf);

        sprintf(buf, "PWM:%3d%% BYP:%3d%%",
                pwm_voltage,
                bypass_duty);
        OLED_P6x8Str(0, 4, (unsigned char *)buf);

        OLED_P6x8Str(0, 6, (unsigned char *)"STATE: TAPER     ");
    }
    else
    {
        i_show = (int)(icharge_current_mA + 0.5f);

        sprintf(buf, "DAY B:%2d.%02dV   ",
                bat_x100 / 100,
                bat_x100 % 100);
        OLED_P6x8Str(0, 0, (unsigned char *)buf);

        sprintf(buf, "I:%4dmA PWM:%3d%%",
                i_show,
                pwm_voltage);
        OLED_P6x8Str(0, 2, (unsigned char *)buf);

        sprintf(buf, "BYP:%3d%%        ",
                bypass_duty);
        OLED_P6x8Str(0, 4, (unsigned char *)buf);

        if (charge_work_mode == CHG_MODE_MANUAL)
        {
            OLED_P6x8Str(0, 6, (unsigned char *)"MODE: MANUAL     ");
        }
        else
        {
            if (charge_state == STATE_STOP)
            {
                OLED_P6x8Str(0, 6, (unsigned char *)"STATE: STOP      ");
            }
            else
            {
                OLED_P6x8Str(0, 6, (unsigned char *)"STATE: NORMAL    ");
            }
        }
    }
}

static void OLED_ShowLEDPage(void)
{
    char buf[32];

    int bat_x100;
    int i_raw;
    int target_int;

    bat_x100 = (int)(vbat_voltage * 100.0f + 0.5f);
    i_raw = (int)(led_current_mA + 0.5f);
    target_int = (int)(led_target_mA + 0.5f);

    sprintf(buf, "NIGHT B:%2d.%02dV",
            bat_x100 / 100,
            bat_x100 % 100);
    OLED_P6x8Str(0, 0, (unsigned char *)buf);

    sprintf(buf, "T:%3dmA I:%4dmA",
            target_int,
            i_raw);
    OLED_P6x8Str(0, 2, (unsigned char *)buf);

    sprintf(buf, "Duty:%3d PB5:%3d%%",
            led_duty,
            pb5_duty);
    OLED_P6x8Str(0, 4, (unsigned char *)buf);

    if (low_bat_lock)
    {
        OLED_P6x8Str(0, 6, (unsigned char *)"LOW BAT LED OFF  ");
    }
    else if (led_work_mode == MODE_AUTO)
    {
        if ((led_current_filter_mA >= led_target_mA - LED_TOL_MA) &&
            (led_current_filter_mA <= led_target_mA + LED_TOL_MA))
        {
            OLED_P6x8Str(0, 6, (unsigned char *)"MODE:AUTO OK     ");
        }
        else if (led_current_filter_mA < led_target_mA - LED_TOL_MA)
        {
            OLED_P6x8Str(0, 6, (unsigned char *)"MODE:AUTO LOW    ");
        }
        else
        {
            OLED_P6x8Str(0, 6, (unsigned char *)"MODE:AUTO HIGH   ");
        }
    }
    else if (led_work_mode == MODE_FLOW)
    {
        OLED_P6x8Str(0, 6, (unsigned char *)"MODE:FLOW        ");
    }
    else
    {
        OLED_P6x8Str(0, 6, (unsigned char *)"MODE:STOP        ");
    }
}

static void UART_PrintInfo(void)
{
    printf("PAGE=%s,BAT=%.3f,SOL=%.3f,CHG_I=%.2f,CHG_If=%.2f,CHG_T=%.2f,CHG_PWM=%u,BYP=%u,CHG_STATE=%d,CHG_MODE=%d,LED_T=%.0f,LED_I=%.0f,LED_F=%.0f,LED_DUTY=%u,PB5=%u,LED_MODE=%d,LOWBAT=%u\r\n",
           (display_page == PAGE_DAY_CHARGE) ? "DAY" : "NIGHT",
           vbat_voltage,
           vsl_voltage,
           icharge_current_mA,
           icharge_filter_mA,
           target_current_mA,
           pwm_voltage,
           bypass_duty,
           charge_state,
           charge_work_mode,
           led_target_mA,
           led_current_mA,
           led_current_filter_mA,
           led_duty,
           pb5_duty,
           led_work_mode,
           low_bat_lock);
}

static void Key_GPIO_Init_User(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOB_CLK_ENABLE();

    GPIO_InitStruct.Pin = KEY_S1_PIN | KEY_S2_PIN | KEY_S3_PIN | KEY_S4_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
}

static uint8_t Key_IsPressed(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
    if (HAL_GPIO_ReadPin(GPIOx, GPIO_Pin) == GPIO_PIN_RESET)
    {
        return 1;
    }

    return 0;
}

static uint8_t Key_ScanEx(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
    uint32_t press_tick;

    if (Key_IsPressed(GPIOx, GPIO_Pin))
    {
        HAL_Delay(20);

        if (Key_IsPressed(GPIOx, GPIO_Pin))
        {
            press_tick = HAL_GetTick();

            while (Key_IsPressed(GPIOx, GPIO_Pin))
            {
                HAL_Delay(10);
            }

            HAL_Delay(20);

            if (HAL_GetTick() - press_tick >= KEY_LONG_MS)
            {
                return KEY_LONG;
            }
            else
            {
                return KEY_SHORT;
            }
        }
    }

    return KEY_NONE;
}

static uint16_t ADC_Read_Channel(uint32_t channel)
{
    ADC_ChannelConfTypeDef sConfig = {0};
    uint16_t adc_value = 0;

    sConfig.Channel = channel;
    sConfig.Rank = ADC_REGULAR_RANK_1;
    sConfig.SamplingTime = ADC_SAMPLETIME_239CYCLES_5;

    if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
    {
        Error_Handler();
    }

    HAL_ADC_Start(&hadc1);

    if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK)
    {
        HAL_ADC_GetValue(&hadc1);
    }

    HAL_ADC_Stop(&hadc1);

    HAL_ADC_Start(&hadc1);

    if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK)
    {
        adc_value = HAL_ADC_GetValue(&hadc1);
    }

    HAL_ADC_Stop(&hadc1);

    return adc_value;
}

static uint16_t ADC_Read_Channel_Average(uint32_t channel, uint8_t times)
{
    uint32_t sum = 0;
    uint8_t i;

    if (times == 0)
    {
        times = 1;
    }

    for (i = 0; i < times; i++)
    {
        sum += ADC_Read_Channel(channel);
        HAL_Delay(1);
    }

    return (uint16_t)(sum / times);
}

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkStruct = {0};
    RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

    RCC_OscStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscStruct.HSEState = RCC_HSE_ON;
    RCC_OscStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
    RCC_OscStruct.HSIState = RCC_HSI_ON;
    RCC_OscStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscStruct.PLL.PLLMUL = RCC_PLL_MUL5;

    if (HAL_RCC_OscConfig(&RCC_OscStruct) != HAL_OK)
    {
        Error_Handler();
    }

    RCC_ClkStruct.ClockType = RCC_CLOCKTYPE_HCLK |
                              RCC_CLOCKTYPE_SYSCLK |
                              RCC_CLOCKTYPE_PCLK1 |
                              RCC_CLOCKTYPE_PCLK2;

    RCC_ClkStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkStruct, FLASH_LATENCY_1) != HAL_OK)
    {
        Error_Handler();
    }

    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
    PeriphClkInit.AdcClockSelection = RCC_ADCPCLK2_DIV4;

    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
    {
        Error_Handler();
    }
}

int fputc(int ch, FILE *f)
{
    HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

void Error_Handler(void)
{
    __disable_irq();

    while (1)
    {
    }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif