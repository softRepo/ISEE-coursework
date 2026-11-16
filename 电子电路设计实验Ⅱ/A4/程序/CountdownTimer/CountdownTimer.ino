#include<EEPROM.h>

int segmentPins[] = {3,2,19,16,18,4,5,17}; // 段选：a,b,c,d,e,f,g,dp
int displayPins[] = {6,15,7,14};           // 四位数码管位选
int times[] = {
    5,10,15,20,30,
    45,100,130,200,230,
    300,400,500,600,700,
    800,900,1000,1500,2000,
    3000
};
int numTimes = 20;  // times[] 的最大下标

byte selectedTimeIndex;
int timerMinute;
int timerSecond;

int buzzerPin = 11;
int aPin = 8;
int bPin = 10;
int buttonPin = 9;
boolean stopped = true;

// 上电初期先屏蔽编码器，避免复位时误计一步。
bool encoderEnabled = false;

// 共阴极段码表，顺序为 a,b,c,d,e,f,g,dp。
byte digits[10][8] = {
    {1,1,1,1,1,1,0,0}, // 0
    {0,1,1,0,0,0,0,0}, // 1
    {1,1,0,1,1,0,1,0}, // 2
    {1,1,1,1,0,0,1,0}, // 3
    {0,1,1,0,0,1,1,0}, // 4
    {1,0,1,1,0,1,1,0}, // 5
    {1,0,1,1,1,1,1,0}, // 6
    {1,1,1,0,0,0,0,0}, // 7
    {1,1,1,1,1,1,1,0}, // 8
    {1,1,1,1,0,1,1,0}  // 9
};

// 结束提示旋律用到的音符频率。
#define NOTE_DO   262
#define NOTE_RE   294
#define NOTE_MI   330
#define NOTE_FA   349
#define NOTE_SOL  392
#define NOTE_LA   440
#define NOTE_SI   494
#define NOTE_DOH  523
#define NOTE_OFF  0

// 倒计时到 00:00 后播放的提示旋律。
int melody[] = {
    NOTE_DO,NOTE_DO,NOTE_SOL,NOTE_SOL,
    NOTE_LA,NOTE_LA,NOTE_SOL,
    NOTE_FA,NOTE_FA,NOTE_MI,NOTE_MI,
    NOTE_RE,NOTE_RE,NOTE_DO
};
int noteDur[] = {
    200,200,200,200,
    200,200,400,
    200,200,200,200,
    200,200,400
};
int melodyLen = sizeof(melody) / sizeof(melody[0]);

void setup()
{
    for(int i=0;i<8;i++) pinMode(segmentPins[i],OUTPUT);
    for(int i=0;i<4;i++) pinMode(displayPins[i],OUTPUT);
    pinMode(buzzerPin,OUTPUT);
    pinMode(buttonPin,INPUT);
    pinMode(aPin,INPUT_PULLUP);
    pinMode(bPin,INPUT_PULLUP);

    // 从 EEPROM 恢复上次选择的时间档位。
    selectedTimeIndex = EEPROM.read(0);

    // 防止 EEPROM 初值或异常值越界。
    if(selectedTimeIndex < 0 || selectedTimeIndex > numTimes)
        selectedTimeIndex = 0;

    // 将时间档位换算成分钟和秒。
    timerMinute = times[selectedTimeIndex] / 100;
    timerSecond = times[selectedTimeIndex] % 100;

    // 稍作延时后再允许编码器输入。
    delay(300);
    encoderEnabled = true;
}

void loop()
{
    if(!digitalRead(buttonPin))
    {
        stopped = !stopped;
        digitalWrite(buzzerPin,LOW);
        while(!digitalRead(buttonPin)){}
    }

    // 运行状态下倒计时归零，播放结束提示音。
    if(timerMinute == 0 && timerSecond == 0 && !stopped)
    {
        playMelody();
        stopped = true;
        timerMinute = times[selectedTimeIndex] / 100;
        timerSecond = times[selectedTimeIndex] % 100;
    }

    updateDisplay();
}

void playMelody()
{
    for(int i=0; i<melodyLen; i++)
    {
        if(melody[i] == NOTE_OFF)
        {
            digitalWrite(buzzerPin, LOW);
        }
        else
        {
            tone(buzzerPin, melody[i]);
        }
        delay(noteDur[i]);
        noTone(buzzerPin);
        delay(30);
    }
    digitalWrite(buzzerPin,LOW);
}

void updateDisplay()
{
    int d[4];
    d[0] = timerMinute / 10;
    d[1] = timerMinute % 10;
    d[2] = timerSecond / 10;
    d[3] = timerSecond % 10;

    for(int i=0;i<4;i++)
    {
        setDigit(i);
        setSegments(d[i]);
        process();
    }
    setDigit(5);
}

void process()
{
    for(int i=0;i<100; i++)
    {
        int change = getEncoderTurn();
        if(stopped)
        {
            changeSetTime(change);
        }
        else
        {
            updateCountingTime();
        }
    }
}

void changeSetTime(int change)
{
    if(change == 0) return;

    selectedTimeIndex += change;

    // 时间档位首尾循环。
    if(selectedTimeIndex < 0)
        selectedTimeIndex = numTimes;
    else if(selectedTimeIndex > numTimes)
        selectedTimeIndex = 0;

    // 保存当前档位，供下次上电恢复。
    EEPROM.write(0, selectedTimeIndex);

    timerMinute = times[selectedTimeIndex] / 100;
    timerSecond = times[selectedTimeIndex] % 100;
}

void updateCountingTime()
{
    static unsigned long lastMillis;
    unsigned long m = millis();
    if(m > (lastMillis + 1000) && (timerSecond > 0 || timerMinute > 0))
    {
        digitalWrite(buzzerPin,HIGH);
        delay(10);
        digitalWrite(buzzerPin,LOW);

        if(timerSecond == 0)
        {
            timerSecond = 59;
            timerMinute--;
        }
        else
        {
            timerSecond--;
        }
        lastMillis = m;
    }
}

void setDigit(int digit)
{
    for(int i=0;i<4;i++)
    {
        digitalWrite(displayPins[i],(digit==i));
    }
}

void setSegments(int n)
{
    for(int i=0;i<8; i++)
    {
        digitalWrite(segmentPins[i],digits[n][i]);
    }
}

int getEncoderTurn()
{
    if(!encoderEnabled) return 0;

    static int oldA = LOW;
    static int oldB = LOW;
    int result = 0;
    int newA = digitalRead(aPin);
    int newB = digitalRead(bPin);

    if(newA != oldA || newB != oldB)
    {
        if(oldA == LOW && newA == HIGH) result = -(oldB * 2 - 1);
    }
    oldA = newA;
    oldB = newB;
    return result;
}
