#ifndef CMT2119A_H
#define CMT2119A_H
#include <stdint.h>
#include <ch32fun.h>
#include "CMT2119A_TWI.h"
#include <math.h>
#include "ch32fun.h"
#include "ch32v003hw.h"

//based on https://github.com/g4eml/RP2040_Synth/blob/main/Arduino/RP2040Synth/CMT2119A.ino by G4EML

typedef struct {
    uint16_t r1; //prescaler2
    uint16_t r6; //prescaler15
    uint16_t r7; //PLL low bits
    uint16_t r8; //PLL high bits
} CMT2119_freq_set_t;

//probably need to refine interface a little

void CMT2119A_init(uint8_t clk_pin, uint8_t dat_pin, double freq);
void CMT2119A_setDefault(void);
void CMT2119A_update(void);
void CMT2119A_updateFreqOnly(CMT2119_freq_set_t* regs);
double CMT2119A_getSetFrequency(void);
void CMT2119A_calcFrequency(double freq, CMT2119_freq_set_t* regs);
void CMT2119A_setFrequency(CMT2119_freq_set_t* regs);

#endif