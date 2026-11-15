#ifndef __BSP_KEY_H
#define __BSP_KEY_H
#include "stm32f1xx_hal.h"

#define S1      HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_0)  //KEY0按键PB0
#define S2      HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_1)  //KEY1按键PB1
#define S3      HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_10)  //KEY2按键PB10
#define S4     HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_11)  //WKUP按键PB11

#define S1_PRES	1	//S1
#define S2_PRES	2	//S2
#define S3_PRES	3	//S3
#define S4_PRES	4	//S4

uint8_t KEY_Scan(uint8_t mode);
#endif