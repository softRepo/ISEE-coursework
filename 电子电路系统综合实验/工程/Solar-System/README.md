# Solar Lamp System

STM32F103C8T6 太阳能景观灯控制器固件。

## 目录

- `Core/`：主控制程序、外设初始化、OLED 驱动、按键和字库。
- `Drivers/`：STM32F1 HAL 与 CMSIS。
- `MDK-ARM/`：Keil 工程和 ARM 汇编启动文件。
- `cmake/`：GNU Arm 工具链与 STM32CubeMX 构建配置。
- `Solar-System.ioc`：STM32CubeMX 外设与引脚配置。

## 编译

使用 Keil µVision 5、ARM Compiler 5 和 `Keil.STM32F1xx_DFP 2.2.0`，打开 [Solar-System.uvprojx](MDK-ARM/Solar-System.uvprojx)，选择 `Solar-System` 目标并执行 Build。编译输出位于 `MDK-ARM/Solar-System/`。

也可使用 CMake 3.22 或更高版本、Ninja 和 GNU Arm Embedded Toolchain。在本目录运行，工具链的 `arm-none-eabi-*` 命令需位于 PATH 中：

```sh
cmake --preset Debug
cmake --build --preset Debug
```

CMake 输出位于 `build/Debug/`。

## 主要接口

| 接口 | 引脚 | 用途 |
| --- | --- | --- |
| ADC1 | PA1、PA2 | 储能端与充电输入电压采样 |
| ADC1 | PA3、PA4 | LED 与充电电流采样 |
| TIM3 CH1、CH2 | PB4、PB5 | 充电与 LED PWM |
| I2C1 | PB6、PB7 | OLED 显示 |
| GPIO | PB0、PB1、PB10、PB11 | 四个独立按键 |

控制参数位于 [Core/Src/main.c](Core/Src/main.c)。硬件电路、标定方法和测试结果见[实验报告](../../实验报告.pdf)。

第三方组件的原始版权声明保留在源码中，许可文件分别见 [STM32F1 HAL](Drivers/STM32F1xx_HAL_Driver/LICENSE.md)、[CMSIS Core](Drivers/CMSIS/LICENSE.txt) 和 [CMSIS Device F1](Drivers/CMSIS/Device/ST/STM32F1xx/License.md)。
