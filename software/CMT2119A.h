#ifndef CMT2119A_H
#define CMT2119A_H
#include <stdint.h>
#include <ch32fun.h>
#include <stdbool.h>
#include "CMT2119A_TWI.h"
#include <math.h>
#include "ch32fun.h"
#include "ch32v003hw.h"

//based on https://github.com/g4eml/RP2040_Synth/blob/main/Arduino/RP2040Synth/CMT2119A.ino by G4EML

typedef struct {
    uint32_t freq;
    bool prescale2;
    bool prescale15;
    uint16_t pll_low;
    uint16_t pll_high;
} CMT2119_freq_set_t;

enum CMT2119A_modulation{
    MOD_OOK,
    MOD_FSK,
    MOD_GFSK
};

enum CMT2119A_low_off_time{
    OFF_20MS,
    OFF_30MS,
    OFF_40MS,
    OFF_50MS,
    OFF_60MS,
    OFF_70MS,
    OFF_80MS,
    OFF_90MS
};

typedef struct {
    uint32_t freq_out_hz;
    uint32_t fsk_dev_hz;
    uint32_t gfsk_rate_bps;
    int power_output_dbm;
    enum CMT2119A_modulation modulation;
    enum CMT2119A_low_off_time off_time;
    uint16_t pa_ramp_time;
    bool rising_edge_start;
    bool invert_symbols;
    bool xo_current_boost;
} CMT2119A_settings_t;

//probably need to refine interface a little

void CMT2119A_init(uint8_t clk_pin, uint8_t dat_pin, CMT2119A_settings_t *set);

void CMT2119A_setPowerOut(int dbm, bool ook_en);
void CMT2119A_setModulation(enum CMT2119A_modulation modulation);
void CMT2119A_setLowOffTime(enum CMT2119A_low_off_time off_time);
void CMT2119A_setPArampTime(uint16_t us);
void CMT2119A_setRisingEdgeStart(bool rising);
void CMT2119A_setSymbolInversion(bool inv);
void CMT2119A_setCrystalCurrentBoost(bool boost);
void CMT2119A_setGFSKrate(uint32_t bps);
void CMT2119A_setFrequencyDev(uint32_t freq, uint32_t deviation);

void CMT2119A_update(void);
void CMT2119A_updateFreqOnly();
void CMT2119A_updateDeviationOnly();
double CMT2119A_getSetFrequency(void);

void CMT2119A_calcFrequency(uint32_t freq, CMT2119_freq_set_t* freq_set);
void CMT2119A_calcDeviation(uint32_t deviation, uint16_t *div, CMT2119_freq_set_t* freq_set);
void CMT2119A_setFrequencyFromStruct(CMT2119_freq_set_t* freq_set);

#endif