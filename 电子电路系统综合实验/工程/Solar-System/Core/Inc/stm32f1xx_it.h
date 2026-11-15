/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    stm32f1xx_it.h
  * @brief   This file contains the headers of the interrupt handlers.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F1xx_IT_H
#define __STM32F1xx_IT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */
  //3.1 计数信号
  extern volatile uint16_t CountTimer;
  extern volatile uint16_t CountTimer10ms;
  extern volatile uint16_t CountTimer500ms;
  extern volatile uint16_t CountTimer10s;
  extern volatile uint16_t DelayADC;
  extern volatile uint16_t CountLight;
  //3.2 控制循环的信号
  extern volatile uint8_t flagLight;
  extern volatile uint8_t flag_KS; //按键扫描请求，0没有请求
  extern uint8_t KeyV; //按键
  extern volatile uint8_t ADC_flag;		// ADC采样标志，默认0未就绪
  extern volatile uint16_t OLEDFlash_flag; // OLED刷新标志，默认不刷新
  extern volatile uint16_t flag_test;		// 测试标志
  extern uint8_t sys_flag;		//蓝牙控制：总开关（还没加）

  extern volatile uint8_t display_state; // 声明一个外部变量用于显示状态机
  extern volatile uint16_t splash_timer_ms;
/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void SVC_Handler(void);
void DebugMon_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);
void TIM1_UP_IRQHandler(void);
void USART1_IRQHandler(void);
/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F1xx_IT_H */
