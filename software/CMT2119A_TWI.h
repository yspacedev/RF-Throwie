#ifndef TWI_H
#define TWI_H

#include <ch32fun.h>
#include <stdio.h>
#include <stdint.h>

//based on https://github.com/g4eml/RP2040_Synth/blob/main/Arduino/RP2040Synth/CMT2119A.ino by G4EML

extern uint8_t TWIDAT;   // match whatever type you actually declared them as
extern uint8_t TWICLK;

#define TWI_CYCLE_DELAY {Delay_Us(1);}
#define TWI_MS_DELAY {Delay_Ms(1);}

#define TWI_DAT_OUT {funPinMode(TWIDAT, FUN_OUTPUT);}
#define TWI_DAT_IN {funPinMode(TWIDAT, FUN_INPUT);}
#define TWI_DAT_READ (funDigitalRead(TWIDAT))
#define TWI_DAT_HIGH {funDigitalWrite(TWIDAT, FUN_HIGH);}
#define TWI_DAT_LOW {funDigitalWrite(TWIDAT, FUN_LOW);}
#define TWI_CLK_OUT {funPinMode(TWICLK, FUN_OUTPUT);}
#define TWI_CLK_HIGH {funDigitalWrite(TWICLK, FUN_HIGH);}
#define TWI_CLK_LOW {funDigitalWrite(TWICLK, FUN_LOW);}

void TWI_init(uint8_t clk_pin, uint8_t dat_pin);
void TWI_Write(uint8_t x);
uint8_t TWI_Read(void);
void TWI_WRREG(uint8_t addr, uint8_t data);
uint8_t TWI_RDREG(uint8_t addr);
void TWI_RAM1(uint8_t addr, uint16_t data);
void TWI_RAM(const uint32_t *x, uint8_t n);
void TWI_reset(void);
void CMT2119A_TWI_OFF(void);
void CMT2119A_SOFT_RST(void);
void CMT2119A_RESET(void);
void TWI_EEPROM_ERASE(uint8_t add);
void TWI_EEPROM_WRITE(uint8_t add, uint16_t dat);
uint16_t TWI_EEPROM_READ(uint8_t add);
void TWI_EEPROM_SETUP(void);
void TWI_EEPROM_END(void);
void CMT2119A_EEPROM_BURN(uint16_t* regmap);

#endif