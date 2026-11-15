/* Function List:
*
* 1. void delay(unsigned int z) -- ��ʱ����,����
* 2. void IIC_Start() -- ����I2C����
* 3. void IIC_Stop() -- �ر�I2C����
* 4. void Write_IIC_Byte(unsigned char IIC_Byte) -- ͨ��I2C����дһ��byte������
* 5. void OLED_WrDat(unsigned char dat) -- ��OLED��д����
* 6. void OLED_WrCmd(unsigned char cmd) -- ��OLED��д����
* 7. void OLED_Set_Pos(unsigned char x, unsigned char y) -- ������ʾ����
* 8. void OLED_Fill(unsigned char bmp_dat) -- ȫ����ʾ(��ʾBMPͼƬʱ�Ż��õ��˹���)
* 9. void OLED_CLS(void) -- ��λ/����
* 10. void OLED_Init(void) -- OLED����ʼ�����򣬴˺���Ӧ�ڲ�����Ļ֮ǰ���ȵ���
* 11. void OLED_P6x8Str(unsigned char x, y,unsigned char ch[]) -- 6x8������������ʾASCII�����С���У���̫����
* 12. void OLED_P8x16Str(unsigned char x, y,unsigned char ch[]) -- 8x16������������ʾASCII�룬�ǳ�����
* 13.void OLED_P16x16Ch(unsigned char x, y, N) -- 16x16������������ʾ���ֵ���С���У������ø������塢�Ӵ֡���б���»��ߵ�
* 14.void Draw_BMP(unsigned char x0, y0,x1, y1,unsigned char BMP[]) -- ��128x64���ص�BMPλͼ��ȡ�����������ֱ�Ȼ���Ƶ�codetab�У��˺������ü���
*
* History: none;
*
*************************************************************************************/

#ifndef _OLED12864_H_
#define _OLED12864_H_
#include "main.h"
#include "gpio.h"
#include "codetab.h"
#include "stm32f1xx_it.h"

#define	Brightness	0xCF
#define X_WIDTH 	128
#define Y_WIDTH 	64
/*********************OLED���������õ���ʱ����************************************/
void delay(unsigned int z);
void IIC_GPIO_Init(void);
void OLED_WrDat(unsigned char IIC_Data);
void OLED_WrCmd(unsigned char IIC_Command);
void OLED_Set_Pos(unsigned char x, unsigned char y);
void OLED_Fill(unsigned char bmp_dat);
void OLED_Clear(void);
void OLED_Init(void);
void OLED_ON(void);
void OLED_OFF(void);
void OLED_ColorTurn(unsigned char i);
void OLED_DisplayTurn(unsigned char i);
void OLED_P6x8Str(unsigned char x, unsigned char y,unsigned char ch[]);
void OLED_P8x16Char(unsigned char x, unsigned char y,unsigned char ch);
void OLED_P8x16NUM(unsigned char x, unsigned char y,unsigned char NUM);
void Draw_BMP(unsigned char x0, unsigned char y0,unsigned char x1, unsigned char y1,const char BMP[]);
void OLED_ShowStr_HZ(unsigned char x, unsigned char y, unsigned char *str);
void OLED_P16X32Str(unsigned char x, unsigned char y,unsigned char *str);

void OLED_Scroll_LastLine(const char *str);
void OLED_Scroll_Stop(void);
void Clear_LastLine(void);

void OLED_Scroll_FirstLine(const char *str, unsigned char font_size);
void Clear_FirstLine(void);                   // 清除第一行
void OLED_P6x8Char(unsigned char x, unsigned char y, unsigned char ch); // 显示单个字符
void OLED_Clear_Part(unsigned char start_page);

// 在OLED12864.h的函数声明区域添加
void OLED_DrawStartupImage_Overlay(void);

#endif
