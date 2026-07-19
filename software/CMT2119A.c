#include "CMT2119A.h"

//TODO: improve interface (how to calculate frequency, configure internal settings variable, how to pass frequency set struct around)
//TODO: add more configuration as well as FSK capabilities
//TODO: find good default configuration
//TODO: add options to select default config


//based on https://github.com/g4eml/RP2040_Synth/blob/main/Arduino/RP2040Synth/CMT2119A.ino by G4EML

uint16_t CMT2119Asettings[21];

double refosc = 26.0; //MHz

//915MHz OOK, +13dBm power, DATA rising to transmit, 0us PA ramp, stop with 20ms DATA low
//generated with official CMOSTEK RFPDK
static const uint16_t CMT2119defaultOOK[21] = {
0x007F,
0x5000, //prescale2
0x0000, 
0x0000,
0x0000,
0xF000,
0x0000, //prescale2
0x6276, //PLL divider low
0x4600, //PLL divider high
0x0000, //FSK deviation related
0x2401,
0x01B0,
0x8000,
0x0006,
0xFFFF,
0x0020,
0x5F1E,
0x22D6,
0x0E13,
0x0019,
0x2000,
};

//915MHz, FSK, 2.4ksps, 12.5kHz deviation, +13dBm
static const uint16_t CMT2119defaultFSK[21] = {
0x007F,
0x5000, //prescale2
0x0000,
0x0000,
0x0000,
0xF000,
0x0000, //prescale15
0x6276, //PLL divider low
0x4608, //PLL divider high
0x00B0, //FSK deviation?
0x2401,
0x0081,
0x8000,
0x0000,
0xFFFF,
0x0020,
0x5F1E,
0xA2D6,
0x0E13,
0x0019,
0x0000,
};

//431.1MHz, GFSK, 2.4ksps, 1kHz deviation, +13dBm
static const uint16_t CMT2119defaultGFSK[21] = {
0x007F,
0x5400,
0x0000,
0x0000,
0x0000,
0xF000,
0x0000,
0x7A17,
0x4208,
0x000F,
0xA401,
0x11A0,
0x8000,
0x0000,
0xFFFF,
0x0020,
0x5FD9,
0xA2D6,
0x0E13,
0x0019,
0x0000,
};


void CMT2119A_init(uint8_t clk_pin, uint8_t dat_pin, double freq){
    TWI_init(clk_pin, dat_pin);
    for(int i = 0;i<21;i++){
        CMT2119Asettings[i] = CMT2119defaultGFSK[i];
    }
    if (freq<=150.0 || freq>1297.0){ //if outside of valid frequency range
        CMT2119A_update();
    } else {
        CMT2119_freq_set_t f;
        CMT2119A_calcFrequency(freq, &f);
        CMT2119A_setFrequency(&f);
        CMT2119A_update();
    }
}

void CMT2119A_setDefault(void){
    for(int i = 0;i<21;i++){
        CMT2119Asettings[i] = CMT2119defaultGFSK[i];
    }
    CMT2119A_update();
}

void CMT2119A_update(void){
    TWI_reset(); //step 1
    TWI_WRREG(0x3d, 0x01); //step 2 send SOFT_RST
    TWI_MS_DELAY
    TWI_MS_DELAY

    //some proprietary command preamble from the datasheet
    TWI_WRREG(0x02, 0x78); //Open LDO & Osc step 3

    TWI_WRREG(0x2F, 0x80); //vActiveRegsister step 4
    TWI_WRREG(0x35, 0xCA);
    TWI_WRREG(0x36, 0xEB);
    TWI_WRREG(0x37, 0x37);
    TWI_WRREG(0x38, 0x82);

    TWI_WRREG(0x12, 0x10); //vEnableRegMode step 5
    TWI_WRREG(0x12, 0x00);
    TWI_WRREG(0x24, 0x07);
    TWI_WRREG(0x1D, 0x20);

    //program the default RAM config by RFPDK generated setup

    for(int i =0;i<21;i++){
        TWI_RAM1(i,CMT2119Asettings[i]);
    }

    TWI_WRREG(0x0D, 0x02); //step 7 send the TWI_OFF command. Control reverts to simple DAT signals

    TWI_DAT_HIGH //put into transmit state with low output
    TWI_CYCLE_DELAY
    TWI_DAT_LOW
}

void CMT2119A_updateFreqOnly(CMT2119_freq_set_t* regs){
    TWI_reset(); //step 1
    TWI_MS_DELAY
    TWI_MS_DELAY

    //should I use the values in the settings array or allow them to be used as parameters?
    //or pass in a struct pointer?
    TWI_RAM1(7,regs->r7);       //just update the divider registers. 
    TWI_RAM1(8,regs->r8);

    TWI_WRREG(0x0D, 0x02); //step 7 send the TWI_OFF command. Control reverts to simple DAT signals

    TWI_DAT_HIGH //put into transmit state
    TWI_CYCLE_DELAY
    TWI_DAT_LOW
}

double CMT2119A_getSetFrequency(void){
    uint8_t prescale15;
    uint8_t prescale2;
    uint32_t divider;
    double vco;
    double pfd = refosc/131072.0;
    double diva;

    prescale15 = CMT2119Asettings[6] & 0x01;
    prescale2 = (CMT2119Asettings[1] & 0x0400) >> 10;
    divider = ((CMT2119Asettings[8] & 0xFF00) << 8) + CMT2119Asettings[7]; 

    vco = (double) divider * pfd;
    diva=1;
    if(prescale2) diva = diva * 2;
    if(prescale15) diva = diva * 1.5;
    
    return vco/diva;
}

void CMT2119A_calcFrequency(double freq, CMT2119_freq_set_t* regs){
    double pfd = refosc/131072.0;
    uint8_t prescale15;
    uint8_t prescale2;

    if(freq<=320.0){
        prescale15=1;
        prescale2=1;
    }else if(freq<=480.0){
        prescale15=0;
        prescale2=1;
    }else if(freq<=640.0){
        prescale15=1;
        prescale2=0;
    }else{
        prescale15=0;
        prescale2=0;
    }

    regs->r6 = prescale15;
    regs->r1 = prescale2 ? 0x5400:0x5000;

    if(prescale15) freq = freq * 1.5;
    if(prescale2) freq = freq * 2.0;

    //frequency
    uint32_t pll = round((freq/pfd)/2) *2;                //round to nearest even number
    regs->r7 = pll & 0xfffe;              //lsb must always be zero. 
    uint16_t pllh = (pll >> 8) & 0xFF00;
    regs->r8 = pllh;
    //CMT2119Asettings[9] = 0;
}

void CMT2119A_setFrequency(CMT2119_freq_set_t* regs){
    CMT2119Asettings[1] = regs->r1;
    CMT2119Asettings[6] = regs->r6;
    CMT2119Asettings[7] = regs->r7;
    CMT2119Asettings[8] = regs->r8;
}