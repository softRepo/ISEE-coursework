#include "BSP_KEY.h"

//按键处理函数
//返回按键值
uint8_t KEY_Scan(uint8_t mode)
{
    static uint8_t key_up=1;//按键松开标志
    if(mode) key_up=1; //支持连按
    if(key_up&&(S1==0||S2==0||S3==0||S4==0))
    {
        HAL_Delay(10);//去抖动
        key_up=0;
        if(S1==0)return S1_PRES;
        if(S2==0)return S2_PRES;
        if(S3==0)return S3_PRES;
        if(S4==0)return S4_PRES;
    }else if(S1==1&&S2==1&&S3==1&&S4==1)key_up=1;
    return 0;//无按键按下
}