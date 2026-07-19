#include "CMT2119A_TWI.h"

 uint8_t TWICLK = PC1;
 uint8_t TWIDAT = PC2;

void TWI_init(uint8_t clk_pin, uint8_t dat_pin){
    funGpioInitAll();
    TWICLK = clk_pin;
    TWIDAT = dat_pin;
    TWI_DAT_OUT
    TWI_CLK_OUT
    TWI_DAT_HIGH
    TWI_CLK_HIGH
}


void TWI_Write(uint8_t x){
    TWI_CLK_HIGH
    TWI_DAT_LOW
    for(uint8_t i=0; i<8; ++i){
        TWI_CLK_HIGH
        if(x & 0x80) TWI_DAT_HIGH else TWI_DAT_LOW
        TWI_CYCLE_DELAY
        funDigitalWrite(TWICLK,FUN_LOW); 
        TWI_CYCLE_DELAY
        x<<=1;
    }
    TWI_CLK_HIGH
    TWI_DAT_LOW
}

uint8_t TWI_Read(void){
    uint8_t r=0;
    TWI_DAT_IN
    TWI_CLK_HIGH
    for(uint8_t i=0; i<8; ++i){
        TWI_CLK_HIGH
        TWI_CYCLE_DELAY
        r <<= 1;
        funDigitalWrite(TWICLK,FUN_LOW);
        TWI_CYCLE_DELAY
        if(TWI_DAT_READ) r|=1;
    }
    TWI_CLK_HIGH
    TWI_DAT_OUT
    funDigitalWrite(TWIDAT,FUN_LOW);
    return r;
}


void TWI_WRREG(uint8_t addr, uint8_t data){
    TWI_Write(0x80|(addr&0x3f));
    TWI_Write(data);
}

uint8_t TWI_RDREG(uint8_t addr){
    TWI_Write(0xc0|(addr&0x3f));
    return TWI_Read();
}


void TWI_RAM1(uint8_t addr, uint16_t data){
    TWI_WRREG(0x18,addr);
    TWI_WRREG(0x19,data&0xff);
    TWI_WRREG(0x1A,data>>8);
    TWI_WRREG(0x25, 0x01);
}

void TWI_RAM(const uint32_t *x, uint8_t n){
    for(uint8_t i=0; i<n; ++i){
        TWI_RAM1(i,*x++);
    }
}

void TWI_reset(void){
    TWI_DAT_LOW
    TWI_CLK_HIGH
    TWI_CYCLE_DELAY

    for(uint8_t i=0; i<32; ++i){
	    funDigitalWrite(TWICLK,FUN_LOW);
	    TWI_CYCLE_DELAY
	    TWI_CLK_HIGH
        TWI_CYCLE_DELAY	
	}
    //send TWI_RST
    TWI_Write(0x8D);
    TWI_Write(0x00);
}

void CMT2119A_SOFT_RST(void){
    TWI_Write(0xBD);
    TWI_Write(0x01);
}

void CMT2119A_TWI_OFF(void){
    TWI_Write(0x8D);
    TWI_Write(0x02);
}

void CMT2119A_RESET(void){
    TWI_reset(); //step 1
    TWI_WRREG(0x3d, 0x01); //step 2 send SOFT_RST
    TWI_MS_DELAY
    TWI_MS_DELAY
    TWI_WRREG(0x0D, 0x02); //step 7 send the TWI_OFF command. Control reverts to simple DAT signals
}

void TWI_EEPROM_ERASE(uint8_t add){
    uint8_t resp;
    TWI_WRREG(0x17,add);          //Set the EEPROM Address
    TWI_WRREG(0x16,0x39);           //start the erase
    do                          //wait till the erase has completed
    {
        TWI_MS_DELAY
        resp = TWI_RDREG(0x1F);
    }
    while ((resp & 0x08) == 0); 
    TWI_WRREG(0x16,0x31);           //end the erase
}

void TWI_EEPROM_WRITE(uint8_t add, uint16_t dat){
    uint8_t resp;
    TWI_WRREG(0x17,add);          //Set the EEPROM Address
    TWI_WRREG(0x19,dat & 0xFF);   //Set the EEPROM Low Byte
    TWI_WRREG(0x1A,dat >> 8);     //Set the EEPROM High Byte 
    TWI_WRREG(0x16,0x35);         //start the write
    do {                            //wait till the erase has completed
        TWI_MS_DELAY
        resp = TWI_RDREG(0x1F);
    } while ((resp & 0x08) == 0); 
    TWI_WRREG(0x16,0x31);           //end the write
}

uint16_t TWI_EEPROM_READ(uint8_t add){
    uint8_t resp;
    uint16_t val;
    TWI_WRREG(0x17,add);          //Set the EEPROM Address 
    TWI_WRREG(0x16,0x33);         //start the read
    do {
        TWI_MS_DELAY
        resp = TWI_RDREG(0x1F);
    } while ((resp & 0x08) == 0); 
    val =(TWI_RDREG(0x1C) <<8) + TWI_RDREG(0x1B) ;
    TWI_WRREG(0x16,0x31);           //end the read
    return val;
}

void TWI_EEPROM_SETUP(void){
    TWI_WRREG(0x02,0x3B);
    TWI_WRREG(0x2F,0x80);
    TWI_WRREG(0x3F,0x01);
    TWI_WRREG(0x16,0x31);
    TWI_WRREG(0x35,0xCA);
    TWI_WRREG(0x36,0xEB);
    TWI_WRREG(0x37,0x37);
    TWI_WRREG(0x38,0x82);        
}

void TWI_EEPROM_END(void){
    TWI_WRREG(0x16,0x30);
    TWI_WRREG(0x3F,0x00);
    TWI_WRREG(0x0C,0x27);
    TWI_WRREG(0x2F,0x00);
    TWI_WRREG(0x02,0x7F);
    TWI_WRREG(0x0C,0x00);
    TWI_WRREG(0x3D,0x01);         //SOFT_RESET.  
}

//Burn the Chips built in EEPROM. Sequence copied originally from official programmer and then trimmed. 

void CMT2119A_EEPROM_BURN(uint16_t* regmap){
    printf("Burn and Verify Start\n");
    TWI_reset();
    TWI_reset();
    TWI_EEPROM_SETUP();
    for(int r=0;r<0x15;r++){           //erase and write EEPROM values 0x00 - 0x14 
        TWI_EEPROM_ERASE(r);
        TWI_EEPROM_WRITE(r, regmap[r]);
    }

    for(int r=0;r<0x15;r++){           //verify the EEPROM values 0x00 - 0x14
        uint16_t val = TWI_EEPROM_READ(r);
        if( val!= regmap[r]){
            printf("Verify Error at address = ");
            printf("%#02X", r);
            printf(" Value = ");
            printf("%#02X\n", val);
        }
    }
    TWI_EEPROM_END();
    printf("Burn and Verify Complete\n");

    printf("Resetting from CMT2119A EEPROM"); 
    CMT2119A_RESET();
    TWI_DAT_HIGH
}