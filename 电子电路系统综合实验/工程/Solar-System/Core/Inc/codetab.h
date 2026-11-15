/************************************************************************************
* Description:
* 128*64点阵OLED模块例程,
* 在原基础上删掉了一点没用的数组防止报错
*/

#ifndef _CODETAB_H_
#define _CODETAB_H_

/********** 16*16的汉字点阵 ******************************/
extern const char  F16x16[128*32];	// 最多128个汉字，每个汉字32byte (16x16=16x2x8=32byte)

/********** 6*8的点阵 ************************************/
extern const char  F6x8[92][6];	// 92个字符

/********** 8*16的点阵 ************************************/
extern const char  F8X16[128*16]; // 总共支持128个ASCII字符,每个字符16byte (16x8=16byte)
extern const char  F16X32[128*64];// 总共支持10个大数字/字符,每个字符64byte (32x16=64byte)

// 声明开机画面的数组
extern const char gImage_isee[];

// 定义汉字字模数据结构
typedef struct
{
    char Index[2];	// 汉字内码，占2字节
    char Msk[32];	// 点阵数据，占32字节
} typFNT_GB16;

// 声明汉字库数组
extern const typFNT_GB16 GB_16[];


#endif