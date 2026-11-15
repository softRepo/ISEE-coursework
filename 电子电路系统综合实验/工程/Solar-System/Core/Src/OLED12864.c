/*************************************************************************************
*
* Description: 128*64点阵OLED模块例程
*
* Others: none;
*
* Function List:
*
* 1. void delay(unsigned int z) -- 延时函数，软件延时
* 2. void IIC_Start() -- 启动I2C总线
* 3. void IIC_Stop() -- 关闭I2C总线
* 4. void Write_IIC_Byte(unsigned char IIC_Byte) -- 通过I2C总线写一个byte数据
* 5. void OLED_WrDat(unsigned char dat) -- 向OLED中写入数据
* 6. void OLED_WrCmd(unsigned char cmd) -- 向OLED中写入命令
* 7. void OLED_Set_Pos(unsigned char x, unsigned char y) -- 设置显示坐标
* 8. void OLED_Fill(unsigned char bmp_dat) -- 全屏填充(显示BMP图片时会用到此功能)
* 9. void OLED_CLS(void) -- 复位/清屏
* 10. void OLED_Init(void) -- OLED的初始化函数，此函数应在操作屏幕之前最先调用
* 11. void OLED_P6x8Str(unsigned char x, y,unsigned char ch[]) -- 6x8点阵的方式显示ASCII字符，可显示大小写，但太小了
* 12. void OLED_P8x16Str(unsigned char x, y,unsigned char ch[]) -- 8x16点阵的方式显示ASCII码，非常常用
* 13. void OLED_P16x16Ch(unsigned char x, y, N) -- 16x16点阵的方式显示汉字的点阵，可显示大小写，可以自己改字库，添加删除，但比较麻烦
* 14. void Draw_BMP(unsigned char x0, y0,x1, y1,unsigned char BMP[]) -- 将128x64分辨率的BMP位图取模后得到的数据直接放在codetab中，此函数可以直接使用
*
* History: none;
* 注释乱码太多了，我还原了一点写不下去了，就凑合着上面看看吧x
*************************************************************************************/

#include "main.h"
#include "i2c.h"
#include "OLED12864.h"
#include "codetab.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>


#define	Brightness	0xCF
#define X_WIDTH 	128
#define Y_WIDTH 	64

/*********************OLED���������õ���ʱ����************************************/
void delay(unsigned int z)
{
	unsigned int x,y;
	for(x=z;x>0;x--)
		for(y=110;y>0;y--);
}

/*********************OLEDд����************************************/
void OLED_WrDat(unsigned char IIC_Data)
{
	HAL_I2C_Mem_Write(&hi2c1,0x78,0x40,I2C_MEMADD_SIZE_8BIT,&IIC_Data,1,100);
}
/*********************OLEDд����************************************/
void OLED_WrCmd(unsigned char IIC_Command)
{
	HAL_I2C_Mem_Write(&hi2c1,0x78,0x00,I2C_MEMADD_SIZE_8BIT,&IIC_Command,1,100);
}
/*********************OLED ��������************************************/
void OLED_Set_Pos(unsigned char x, unsigned char y)
{
	OLED_WrCmd(0xb0+y);
	OLED_WrCmd(((x&0xf0)>>4)|0x10);
	OLED_WrCmd((x&0x0f)|0x01);
}
/*********************OLEDȫ��************************************/
void OLED_Fill(unsigned char bmp_dat)
{
	unsigned char y,x;
	for(y=0;y<8;y++)
	{
		OLED_WrCmd(0xb0+y);
		OLED_WrCmd(0x00);
		OLED_WrCmd(0x10);
		for(x=0;x<X_WIDTH;x++)
			OLED_WrDat(bmp_dat);
	}
}
/*********************OLED����************************************/
void OLED_Clear(void)
{
	unsigned char y,x;
	for(y=0;y<8;y++)
	{
		OLED_WrCmd(0xb0+y);   // 设置页地址
		OLED_WrCmd(0x00);     // 设置列起始地址的低4位 (0)
		OLED_WrCmd(0x10);     // 设置列起始地址的高4位 (0)
		for(x=0;x<128;x++)
		{
			OLED_WrDat(0x00); // 写入0来清空
		}
	}
}

/*********************OLED����Ļ************************************/
void OLED_ON(void)
{
	OLED_WrCmd(0X8D);  //���õ�ɱ�
	OLED_WrCmd(0X14);  //������ɱ�
	OLED_WrCmd(0XAF);  //OLED����
}
/*********************OLED����Ļ************************************/
void OLED_OFF(void)
{
	OLED_WrCmd(0X8D);  //���õ�ɱ�
	OLED_WrCmd(0X10);  //�رյ�ɱ�
	OLED_WrCmd(0XAE);  //OLED����
}
/*********************OLED��ɫ��ʾ************************************/
void OLED_ColorTurn(unsigned char i)
{
	if(i==0)
	{
		OLED_WrCmd(0xA6);//������ʾ
	}
	else if(i==1)
	{
		OLED_WrCmd(0xA7);//��ɫ��ʾ
	}
}
/*********************OLED��Ļ��ת180��************************************/
void OLED_DisplayTurn(unsigned char i)
{
	if(i==0)
	{
		OLED_WrCmd(0xC8);//������ʾ
		OLED_WrCmd(0xA1);
	}
	else if(i==1)
	{
		OLED_WrCmd(0xC0);//��ת��ʾ
		OLED_WrCmd(0xA0);
	}
}
/*********************OLED��ʼ��************************************/
void OLED_Init(void)
{
	delay(500);//��ʼ��֮ǰ����ʱ����Ҫ��
	OLED_WrCmd(0xae);//--turn off oled panel
	OLED_WrCmd(0x00);//---set low column address
	OLED_WrCmd(0x10);//---set high column address
	OLED_WrCmd(0x40);//--set start line address  Set Mapping RAM Display Start Line (0x00~0x3F)

	OLED_WrCmd(0x81);//--set contrast control register
	OLED_WrCmd(Brightness); // Set SEG Output Current Brightness

	OLED_WrCmd(0xa1);//--Set SEG/Column Mapping     0xa0���ҷ��� 0xa1����
	OLED_WrCmd(0xc8);//Set COM/Row Scan Direction   0xc0���·��� 0xc8����
	OLED_WrCmd(0xa6);//--set normal display
	OLED_WrCmd(0xa8);//--set multiplex ratio(1 to 64)
	OLED_WrCmd(0x3f);//--1/64 duty
	OLED_WrCmd(0xd3);//-set display offset	Shift Mapping RAM Counter (0x00~0x3F)
	OLED_WrCmd(0x00);//-not offset
	OLED_WrCmd(0xd5);//--set display clock divide ratio/oscillator frequency
	OLED_WrCmd(0x80);//--set divide ratio, Set Clock as 100 Frames/Sec
	OLED_WrCmd(0xd9);//--set pre-charge period
	OLED_WrCmd(0xf1);//Set Pre-Charge as 15 Clocks & Discharge as 1 Clock
	OLED_WrCmd(0xda);//--set com pins hardware configuration
	OLED_WrCmd(0x12);
	OLED_WrCmd(0xdb);//--set vcomh
	OLED_WrCmd(0x40);//Set VCOM Deselect Level
	OLED_WrCmd(0x20);//-Set Page Addressing Mode (0x00/0x01/0x02)
	OLED_WrCmd(0x02);//
	OLED_WrCmd(0x8d);//--set Charge Pump enable/disable
	OLED_WrCmd(0x14);//--set(0x10) disable
	OLED_WrCmd(0xa4);// Disable Entire Display On (0xa4/0xa5)
	OLED_WrCmd(0xa6);// Disable Inverse Display On (0xa6/a7)
	OLED_WrCmd(0xaf);//--turn on oled panel
	OLED_Fill(0x00); //��ʼ����
	OLED_Set_Pos(0,0);
}


/*****������������ʾ8*16��һλASCII�ַ�����ʾ�����꣨x,y����yΪҳ��Χ0��7****************/
void OLED_P8x16Char(unsigned char x, unsigned char y,unsigned char ch)
{
	unsigned char i=0;
	unsigned char c =ch-32;
	OLED_Set_Pos(x,y);
	for(i=0;i<8;i++)
		OLED_WrDat(F8X16[16*c+i]);
	OLED_Set_Pos(x,y+1);
	for(i=0;i<8;i++)
		OLED_WrDat(F8X16[16*c+i+8]);
}

/****������������ʾ16*16�����ֻ�8*16 ASCII�ַ�����ʾ�����꣨x,y����
               x��Χ0~119��yΪҳ��Χ0��7**********************/
void OLED_ShowStr_HZ(unsigned char x, unsigned char y, unsigned char *str)
{
	unsigned int k=0;
	unsigned char j;
	while(*str)		//����δ����
	{
		if(*str<128)			//ASCII�ַ�
		{
			if(x>120){x=0;y+=2;}		//128-8=120��һ�е���󣬲���һ���ַ����ȣ��л�����һ�У��ӵ�һ�п�ʼ��ʾ
			OLED_P8x16Char(x,y,*str);
			x=x+8;
			str++;
		}
		else						//���ĺ���
		{
			if(x>112){x=0;y+=2;}		//128-16=112��һ�е���󣬲���һ�����ֿ��ȣ��л�����һ�У��ӵ�һ�п�ʼ��ʾ
			for(k=0;k<3550;k++) 	//�������к��ֿ⣬����Ա�
			{
				if((GB_16[k].Index[0]==*(str))&&(GB_16[k].Index[1]==*(str+1)))	//����Աȳɹ�
				{
					OLED_Set_Pos(x,y);		//����x���꣬����ҳ��ַ����ʾ���ֵ��ϰ벿��
					for(j=0;j<16;j++)
					{
						OLED_WrDat(GB_16[k].Msk[j]);
					}
					OLED_Set_Pos(x,y+1);	//����x���꣬ҳ��1����ʾ�����°벿��
					for(j=0;j<16;j++)
					{
						OLED_WrDat(GB_16[k].Msk[j+16]);
					}
					x=x+16;
					break;  //���ҵ���Ӧ�����ֿ������˳�����������һ������
				}
			}
			str+=2;			//ÿ�����ֶ�Ӧ�����ַ���������ָ���2
		}
	}
}
/****������������ʾ��ʾBMPͼƬ128��64��ʼ������(x,y),x�ķ�Χ0��127��yΪҳ�ķ�Χ0��7*****************
*****sizex��ͼƬ�Ŀ��ȷ�Χ1��128��sizeyΪͼƬ�ĸ߶ȷ�Χ1��64*****************
*/
void Draw_BMP(unsigned char x, unsigned char y,unsigned char sizex, unsigned char sizey,const char BMP[])
{
	unsigned int j=0;
	unsigned char dx,dy;

	if(sizey % 8==0) 			//��ͼƬ�ĸ߶�ת����ҳ����ÿ8����һҳ
		sizey = sizey/8;
	else
		sizey = sizey/8+1;	//�����������ҳ�Ͷ��1.���Ƕ����ҲҪ��һҳ����
	for(dy=0;dy<sizey;dy++)	//��ҳȥд
	{
		OLED_Set_Pos(x,y+dy);	//
		for(dx=0;dx<sizex;dx++)
		{
			OLED_WrDat(BMP[j++]);
		}
	}
}


/***************������������ʾ6*8һ���׼ASCII�ַ��� ��ʾ�����꣨x,y����yΪҳ��Χ0��7****************/
void OLED_P6x8Str(unsigned char x, unsigned char y,unsigned char ch[])
{
	unsigned char c=0,i=0,j=0;
	while (ch[j]!='\0')
	{
		c =ch[j]-32;
		if(x>122){x=0;y++;}//128-6=122��һ�е���󣬲���һ���ַ����ȣ��л�����һ�У��ӵ�һ�п�ʼ��ʾ
		OLED_Set_Pos(x,y);
		for(i=0;i<6;i++)
			OLED_WrDat(F6x8[c][i]);
		x+=6;
		j++;
	}
}


void OLED_P16X32Str(unsigned char x, unsigned char y,unsigned char *str)
{
	unsigned char i=0;
	unsigned char c;
	while(*str)
	{
		c =*str-32;
		OLED_Set_Pos(x,y);
		for(i=0;i<16;i++)
			OLED_WrDat(F16X32[c*64+i]);
		OLED_Set_Pos(x,y+1);
		for(i=0;i<16;i++)
			OLED_WrDat(F16X32[c*64+i+16]);
		OLED_Set_Pos(x,y+2);
		for(i=0;i<16;i++)
			OLED_WrDat(F16X32[c*64+i+32]);
		OLED_Set_Pos(x,y+3);
		for(i=0;i<16;i++)
			OLED_WrDat(F16X32[c*64+i+48]);
		x+=16;
		str++;
	}
}

/***** 以下是为了实现滚动效果写的新函数 *****/


/***** 清除第一行 *****/
void Clear_FirstLine(void)
{
    OLED_Set_Pos(0, 0);  // 设置到第一行（页0）
    for(int x = 0; x < 128; x++) {
        OLED_WrDat(0x00);  // 写入空数据，清除该行
    }
}


/***** 停止滚动并清除第一行 *****/
void OLED_Scroll_Stop(void)
{
    // 通过向滚动函数传递一个空字符串来重置并停止它
    OLED_Scroll_FirstLine("",8);
    Clear_FirstLine();
}


void OLED_Clear_Part(unsigned char start_page)
{
	unsigned char y, x;
	// 只遍历从 start_page 到第7页
	for (y = start_page; y < 8; y++)
	{
		OLED_Set_Pos(0, y); // 设置到每一页的开头
		for (x = 0; x < 128; x++)
		{
			OLED_WrDat(0x00); // 写入空数据
		}
	}
}



/********************************************************************************
 * @brief  在指定位置显示6x8字符串，不自动换行（高效改进版）
 * @param  x   起始X坐标 (可以是负数，用于实现平滑滚入效果)
 * @param  y   起始Y页坐标 (0-7)
 * @param  ch  要显示的字符串 (const unsigned char* 类型以兼容多种字符串)
 * @note   此版本借鉴了原生OLED_P6x8Str函数的效率（每个字符只定位一次），
 *         同时增加了对屏幕左右边界的精确裁剪，非常适合用于实现流畅的滚动显示。
 *         它解决了旧版本中为每一列都调用OLED_Set_Pos而导致的效率低下问题。
 ********************************************************************************/
void OLED_P6x8Str_NoWrap(int x, unsigned char y, const unsigned char ch[])
{
	unsigned char c, i = 0, j;
	int current_x = x;

	// 遍历字符串中的每个字符
	while (ch[i] != '\0')
	{
		// --- 裁剪判断与优化 ---

		// 优化1: 如果字符的起始位置已经超出了屏幕右边界(127)，后续字符也必定不可见，可直接退出循环。
		if (current_x >= 128)
		{
			break;
		}

		// 优化2: 如果字符的结束位置在屏幕左边界之外(即整个字符都在屏幕左边)，则跳过该字符，处理下一个。
		// 字符宽度为6像素，所以结束位置是 current_x + 5。
		if (current_x + 5 < 0)
		{
			current_x += 6; // 移动到下一个字符的逻辑位置
			i++;
			continue;
		}

		// --- 高效绘制 ---
		// 程序能执行到这里，说明字符至少有一部分在屏幕上，准备绘制。

		// 计算字符在F6x8字库中的索引 (ASCII码减去32，因为字库从空格' '开始)
		c = ch[i] - 32;

		// 定位光标到将要绘制的第一个可见列。这是本函数效率提升的关键。
		if (current_x < 0)
		{
			// 如果字符的左边部分在屏幕外，我们从屏幕的第0列开始绘制。
			OLED_Set_Pos(0, y);
		}
		else
		{
			// 如果字符完全在屏幕内或只有右边部分被裁剪，从字符的逻辑起始位置开始绘制。
			OLED_Set_Pos(current_x, y);
		}

		// 遍历当前字符的6列像素数据
		for (j = 0; j < 6; j++)
		{
			// 计算当前列在屏幕上的绝对坐标
			int screen_x = current_x + j;
			// 只绘制在屏幕可见范围(0~127)内的列。
			// 对于左侧被裁剪的字符，这个判断会自动跳过屏幕外的列。
			if (screen_x >= 0 && screen_x < 128)
			{
				// 由于我们已经设置了起始位置，并且OLED控制器在写入数据后会自动将列地址加1，
				// 所以我们只需连续写入所有可见部分的数据即可。
				OLED_WrDat(F6x8[c][j]);
			}
		}

		// 更新到下一个字符的逻辑起始位置
		current_x += 6;
		i++;
	}
}

void OLED_Scroll_FirstLine(const char *str, unsigned char font_size)
{
	// 为不同字体大小维护独立的静态变量
	static int scroll_pos_6x8 = 128, scroll_pos_8x16 = 128;
	static char current_str_6x8[64] = "", current_str_8x16[64] = "";
	static uint32_t last_update_time_6x8 = 0, last_update_time_8x16 = 0;

	// 根据字体大小选择对应的变量
	int* p_scroll_pos;
	char* p_current_str;
	uint32_t* p_last_update_time;
	unsigned char char_width;

	if (font_size == 8) {
		p_scroll_pos = &scroll_pos_8x16;
		p_current_str = current_str_8x16;
		p_last_update_time = &last_update_time_8x16;
		char_width = 8;
	} else { // 默认为6x8字体
		p_scroll_pos = &scroll_pos_6x8;
		p_current_str = current_str_6x8;
		p_last_update_time = &last_update_time_6x8;
		char_width = 6;
	}

	if (str == NULL) str = "";

	// 1. 字符串改变时重置
	if (strcmp(p_current_str, str) != 0) {
		strncpy(p_current_str, str, 63);
		p_current_str[63] = '\0';
		*p_scroll_pos = 128;
	}

	if (p_current_str[0] == '\0') return;

	// 2. 平滑更新滚动位置
	uint32_t current_time = HAL_GetTick();
	const uint32_t scroll_interval = 15;
	if (current_time - *p_last_update_time > scroll_interval) {
		*p_last_update_time = current_time;

		const int scroll_step = 2;
		*p_scroll_pos -= scroll_step;

		int str_width = strlen(p_current_str) * char_width;
		if (*p_scroll_pos < -str_width) {
			*p_scroll_pos = 128;
		}
	}
	// 3. 根据字体大小调用不同的绘图函数
	if (font_size == 8) {
		// 8x16字体从第0页开始，会占用第0和第1页
		OLED_P6x8Str_NoWrap(*p_scroll_pos, 0, (unsigned char*)p_current_str);
	} else {
		OLED_P6x8Str_NoWrap(*p_scroll_pos, 0, (unsigned char*)p_current_str);
	}
}

/**
  * @brief  在屏幕上静态绘制开机图片，不清除屏幕
  * @param  None
  * @retval None
  * @note   此函数设计为在现有内容上“覆盖”显示图片。
  */

void OLED_DrawStartupImage_Overlay(void)
{
	// 1. 从 gImage_isee 数组中解析出图像元数据
	// 假设 gImage_isee 数组已经在此文件中定义
	unsigned char x = gImage_isee[0];       // X坐标
	unsigned char y = gImage_isee[1];       // Y页坐标
	unsigned char sizex = gImage_isee[2];   // 图像宽度
	unsigned char sizey = gImage_isee[4];   // 图像高度

	// 2. 图像数据指针，跳过6字节的头部
	const char *p_bmp_data = (const char*)&gImage_isee[6];

	// 3. 直接在指定位置绘制图像
	Draw_BMP(x, y, sizex, sizey, p_bmp_data);

}