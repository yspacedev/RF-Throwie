#include "CMT2119A.h"
#include "stdbool.h"

//TODO: work on increasing frequency accuracy
//TODO: test if PA ramp setting matters in FSK (doesn't as far as I can tell but I would need to view a time domain waveform to confirm)
//TODO: implement second power LUT for second frequency range
//TODO: find out why reprogramming in GFSK mode can cause spurs that persist through resets but that go away after power cycles
//TODO: implement integer math in frequency calaculations

//based on https://github.com/g4eml/RP2040_Synth/blob/main/Arduino/RP2040Synth/CMT2119A.ino by G4EML

#define EQUIV_REF_FREQ_RECIP ((double)(131072.0/26000000.0))
double refosc = 26.0; //MHz

uint16_t CMT2119Aregs[21];

//only contains reserved bits set, this configuration will do nothing
//since the PLL dividers are not set
static const uint16_t CMT2119Aresetvals[21] = {
    0x007F, //reserved
    0x1000, //rising edge start, prescale2
    0x0000, //reserved
    0x0000, //reserved
    0x0000, //reserved
    0xF000, //reserved
    0x0000, //reserved
    0x0000, //PLL low
    0x0000, //PLL high, FSK enable, tx stop low duration
    0x0000, //FSK deviation
    0x2400, //GFSK enable, invert symbols
    0x0000, //GFSK rate reciprocal
    0x8000, //PA ramp time
    0x0000, //unknown function
    0xFFFF, //reserved
    0x0020, //reserved
    0x4700, //PA power output adjust
    0x22D6, //FSK enable, XO current
    0x0E13, //reserved
    0x0019, //reserved
    0x0000, //reserved
};

//hard to find formulae or derive functions from registers
//both start at -10dBm and continue to 14dBm

//index 0 here corresponds to index 13 for the pwr_field_lut since everything
//before is 0x15
static const uint8_t reg13_lut[] = {
    0x13, 0x12, 
    0x10, 0x0E, 0x0D, 0x0B, 0x0A, 
    0x0A, 0x09, 0x08, 0x06, 0x03,
};

//upper byte of register 16 has a clear band selection with thresholds
//the rest just needs a LUT
static const uint8_t pwr_field_lut[] = {
    0x86, 0x47, 0x48, 0x49, 0x4A,
    0x4B, 0x06, 0x07, 0x08, 0x09,
    0x0A, 0x0B, 0xCC, 0xC7, 0xC8,
    0xC9, 0xCA, 0x0C, 0x0E, 0x10,
    0x12, 0xD4, 0x18, 0x1E, 0xF0,
};

void CMT2119A_setPowerOut(int dbm, bool ook_en){
    if (dbm > 14 || dbm < -10) return;
    int idx = dbm+10;
    int bit11 = (dbm>-5)?1:0;
    int bit12 = (dbm>2)?1:0;
    CMT2119Aregs[16] = CMT2119Aresetvals[16] | bit12<<12 | bit11<<11 | pwr_field_lut[idx];
    if (ook_en) {
        int lut13_idx = idx-13;
        if (lut13_idx<0){
            CMT2119Aregs[13] = CMT2119Aresetvals[13] | 0x15;
        } else {
            CMT2119Aregs[13] = CMT2119Aresetvals[13] | reg13_lut[lut13_idx];
        }
    }
}

void CMT2119A_setModulation(enum CMT2119A_modulation modulation){
    switch (modulation){
        case MOD_OOK:
            CMT2119Aregs[8] &= ~(1<<3);
            CMT2119Aregs[17] &= ~(1<<15);
            break;
        case MOD_FSK:
            CMT2119Aregs[8] |= (1<<3);
            CMT2119Aregs[17] |= (1<<15);
            break;
        case MOD_GFSK:
            CMT2119Aregs[8] |= (1<<3);
            CMT2119Aregs[17] |= (1<<15);
            CMT2119Aregs[10] |= (1<<15);
            break;
    }
}

void CMT2119A_setLowOffTime(enum CMT2119A_low_off_time off_time){
    CMT2119Aregs[8] &= ~0b111;
    CMT2119Aregs[8] |= off_time & 0b111;
}

void CMT2119A_setPArampTime(uint16_t us){
    int reg = 0;
    //even though we only have data for below 128, I'll assume it carries over
    if (us<256){
        reg = ((3*us)>>3) - 2; //floor(3t/8)-2
    } else if (us>=256){
        reg = ((95*us)>>8) - 2; //floor(95t/256)-2
    }

    if (reg<=0) return;

    CMT2119Aregs[12] = CMT2119Aresetvals[12] | (reg & 0x1FF);
}

void CMT2119A_setRisingEdgeStart(bool rising){
    if (rising){
        CMT2119Aregs[1] |= (1<<14);
        CMT2119Aregs[10] |= (1<<0);
    } else {
        CMT2119Aregs[1] &= ~(1<<14);
        CMT2119Aregs[10] &= ~(1<<0);
    }
}

void CMT2119A_setSymbolInversion(bool inv){
    if (inv){
        CMT2119Aregs[10] |= (1<<14);
    } else {
        CMT2119Aregs[10] &= ~(1<<14);
    }
}

void CMT2119A_setCrystalCurrentBoost(bool boost){
    if (boost){
        CMT2119Aregs[17] |= (1<<12) | (1<<10);
    } else {
        CMT2119Aregs[17] &= ~((1<<12) | (1<<10));
    }
}

void CMT2119A_setGFSKrate(uint32_t bps){
    CMT2119Aregs[11] = 0x7FFF & (10769485/bps);
}

void CMT2119A_setFrequencyDev(uint32_t freq, uint32_t deviation){
    CMT2119_freq_set_t f;
    uint16_t reg;
    CMT2119A_calcFrequency(freq, &f);
    CMT2119A_setFrequencyFromStruct(&f);
    CMT2119A_calcDeviation(deviation, &reg, &f);
    CMT2119Aregs[9] = reg & 0x01FF;
}

void CMT2119A_init(uint8_t clk_pin, uint8_t dat_pin, CMT2119A_settings_t *set){
    TWI_init(clk_pin, dat_pin);
    //initialize fields with fixed values
    for(int i = 0;i<21;i++){
        CMT2119Aregs[i] |= CMT2119Aresetvals[i];
    }
    //set fields
    CMT2119A_setPowerOut(set->power_output_dbm, (set->modulation==MOD_OOK));
    CMT2119A_setModulation(set->modulation);
    CMT2119A_setLowOffTime(set->off_time);
    CMT2119A_setPArampTime(set->pa_ramp_time);
    CMT2119A_setRisingEdgeStart(set->rising_edge_start);
    CMT2119A_setSymbolInversion(set->invert_symbols);
    CMT2119A_setCrystalCurrentBoost(set->xo_current_boost);
    CMT2119A_setGFSKrate(set->gfsk_rate_bps);
    CMT2119A_setFrequencyDev(set->freq_out_hz, set->fsk_dev_hz);

    for(int i = 0;i<21;i++){
        if (i>=7 && i <=9){
            printf("register %d: %04x\r\n", i, CMT2119Aregs[i]);
        }
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
        TWI_RAM1(i,CMT2119Aregs[i]);
    }

    TWI_WRREG(0x0D, 0x02); //step 7 send the TWI_OFF command. Control reverts to simple DAT signals

    TWI_DAT_HIGH //put into transmit state with low output
    TWI_CYCLE_DELAY
    TWI_DAT_LOW
}

void CMT2119A_updateFreqOnly(){
    TWI_reset(); //step 1
    TWI_MS_DELAY
    TWI_MS_DELAY

    //should I use the values in the settings array or allow them to be used as parameters?
    //or pass in a struct pointer?
    TWI_RAM1(7,CMT2119Aregs[7]);       //just update the divider registers. 
    TWI_RAM1(8,CMT2119Aregs[8]);

    TWI_WRREG(0x0D, 0x02); //step 7 send the TWI_OFF command. Control reverts to simple DAT signals

    TWI_DAT_HIGH //put into transmit state
    TWI_CYCLE_DELAY
    TWI_DAT_LOW
}

void CMT2119A_updateDeviationOnly(){
    TWI_reset(); //step 1
    TWI_MS_DELAY
    TWI_MS_DELAY

    //should I use the values in the settings array or allow them to be used as parameters?
    //or pass in a struct pointer?
    TWI_RAM1(9,CMT2119Aregs[9]);       //just update the deviation register

    TWI_WRREG(0x0D, 0x02); //step 7 send the TWI_OFF command. Control reverts to simple DAT signals

    TWI_DAT_HIGH //put into transmit state
    TWI_CYCLE_DELAY
    TWI_DAT_LOW
}

//may not be needed
double CMT2119A_getSetFrequency(void){
    uint8_t prescale15;
    uint8_t prescale2;
    uint32_t divider;
    double vco;
    double pfd = refosc/131072.0;
    double diva;

    prescale15 = CMT2119Aregs[6] & 0x01;
    prescale2 = (CMT2119Aregs[1] & 0x0400) >> 10;
    divider = ((CMT2119Aregs[8] & 0xFF00) << 8) + CMT2119Aregs[7]; 

    vco = (double) divider * pfd;
    diva=1;
    if(prescale2) diva = diva * 2;
    if(prescale15) diva = diva * 1.5;
    
    return vco/diva;
}

//NOTE: PLL divider is a 24 bit number likely composed of a 6 bit integer divider and a 16 or 18 bit fractional component. 
//These calculations still show a slight mismatch with the results from RFPDK (only off by a little though)
//It's also likely that the power LUT changes based on frequency band, but I haven't looked into that yet
//also, the change is small, so it shouldn't affect functionality that much
void CMT2119A_calcFrequency(uint32_t freq, CMT2119_freq_set_t* freq_set){
    freq_set->freq=freq;
    uint8_t prescale15;
    uint8_t prescale2;

    if(freq<=320000000){
        prescale15=1;
        prescale2=1;
    }else if(freq<=480000000){
        prescale15=0;
        prescale2=1;
    }else if(freq<=640000000){
        prescale15=1;
        prescale2=0;
    }else{
        prescale15=0;
        prescale2=0;
    }

    freq_set->prescale15 = prescale15;
    freq_set->prescale2 = prescale2;

    if(prescale15) freq = freq + (freq>>2); //freq*1.5
    if(prescale2) freq = freq + freq; //freq*2

    printf("VCO frequency: %d\r\n", freq);

    //frequency
    //could this calculation be made more efficient?
    uint32_t pll = (uint32_t)((double)freq*EQUIV_REF_FREQ_RECIP); //I don't think rounding is needed
    printf("PLL multiplication factor: %u\r\n", pll);
    freq_set->pll_low = pll & 0xfffe;              //lsb must always be zero. 
    uint16_t pllh = (pll >> 8) & 0xFF00;
    freq_set->pll_high = pllh;
}

void CMT2119A_calcDeviation(uint32_t deviation, uint16_t *div, CMT2119_freq_set_t* freq_set){
    if(freq_set->prescale15) deviation = deviation + (deviation>>2); //freq*1.5
    if(freq_set->prescale2) deviation = deviation + deviation; //freq*2

    //frequency
    //could this calculation be made more efficient?
    uint32_t pll = (uint32_t)((double)deviation*EQUIV_REF_FREQ_RECIP);
    *div = pll & 0xFFFF;
}

void CMT2119A_setFrequencyFromStruct(CMT2119_freq_set_t* freq_set){
    if (freq_set->prescale2){
        CMT2119Aregs[1] |= (1<<10);
    } else {
        CMT2119Aregs[1] &= ~(1<<10);
    }
    if (freq_set->prescale15){
        CMT2119Aregs[6] |= (1<<0);
    } else {
        CMT2119Aregs[6] &= ~(1<<10);
    }
    CMT2119Aregs[7] = freq_set->pll_low;
    CMT2119Aregs[8] &= ~(0xFF00);
    CMT2119Aregs[8] |= freq_set->pll_high;
}