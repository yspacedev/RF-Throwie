#include "ch32fun.h"
#include "CMT2119A.h"
#include "CMT2119A_TWI.h"
#include "ch32v003hw.h"

#define TONE_HZ 1000
#define CTR_RELOAD (2000000/TONE_HZ)
#define MORSE_WPM 20
//50 units in the word "PARIS"
//50*WPM is the number of units per minute (1/(50*WPM))*MS_PER_MINUTE
#define MORSE_UNIT_MS ((60*1000)/(50*MORSE_WPM))

#define TX_FREQ 432.1 //MHz

const char* morse_string = "KI5ZHW TEST"; //replace with your own callsign or change TX_FREQ to an ISM band

//possibly rearrange to map better onto ascii
const char* MORSE_ALPHABET[] = {
    ".-",   // A
    "-...", // B
    "-.-.", // C
    "-..",  // D
    ".",    // E
    "..-.", // F
    "--.",  // G
    "....", // H
    "..",   // I
    ".---", // J
    "-.-",  // K
    ".-..", // L
    "--",   // M
    "-.",   // N
    "---",  // O
    ".--.", // P
    "--.-", // Q
    ".-.",  // R
    "...",  // S
    "-",    // T
    "..-",  // U
    "...-", // V
    ".--",  // W
    "-..-", // X
    "-.--", // Y
    "--..", // Z
	"-----",// 0
	".----",// 1
	"..---",// 2
	"...--",// 3
	"....-",// 4
	".....",// 5
	"-....",// 6
	"--...",// 7
	"---..",// 8
	"----.",// 9
};

char upper(char c){
	if (c>='a' && c<='z'){
		return c - 20;
	}
	return c;
}

uint8_t morse_lookup(char c){
	uint8_t idx = 0;
	char up_char = (char)upper((int)c);
	if (up_char >= '0' && up_char <= '9') {
		idx=up_char-'0'+26;
	} else if (up_char >= 'A' && up_char <= 'Z'){
		idx=up_char-'A';
	}
	return idx;
}

void morse_char(char char_in){
	if (char_in==' '){
		Delay_Ms(6*MORSE_UNIT_MS); //6 units instead of 7 since 1 unit delay comes after every dit or dah
		return;
	}
	uint8_t idx = morse_lookup(char_in);
	
	if (idx>=(sizeof(MORSE_ALPHABET)/sizeof(char*))) return;
	char* code = (char*)MORSE_ALPHABET[idx];
	char c = *code;
	while (c!=0){
		int sym = 1;
		if (c=='.'){
			sym=1;
		} else if (c=='-') {
			sym=3;
		}
		funPinMode(TWIDAT, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP_AF); //switch to timer output
		TIM2->CTLR1 |= TIM_CEN; //enable timer
		Delay_Ms(sym*MORSE_UNIT_MS);
		TIM2->CTLR1 &= ~TIM_CEN; //disable timer
		funPinMode(TWIDAT, FUN_OUTPUT); //switch back to GPIO
		TWI_DAT_HIGH //keep high to prevent transmitter going into sleep mode

		code++;
		c=*code;
		Delay_Ms(1*MORSE_UNIT_MS);
	}
	Delay_Ms(2*MORSE_UNIT_MS);
}

void encode_morse(const char *str){
	char c = *str;
	while (c!=0){
		morse_char(c);
		str++;
		c=*str;
	}
}

//use a timer to modulate the FSK into a continuous tone
void init_tone_timer(void){
	//PC2 has alternate remapping to T2CH2

	//enable AFIO and TIM2
	RCC->APB2PCENR |= RCC_APB2Periph_AFIO | RCC_APB2Periph_GPIOC;
	RCC->APB1PCENR |= RCC_APB1Periph_TIM2;

	//remap alternate function
	AFIO->PCFR1 |= AFIO_PCFR1_TIM2_REMAP_PARTIALREMAP1;

	//reset TIM2
	RCC->APB1PRSTR |= RCC_APB1Periph_TIM2;
	RCC->APB1PRSTR &= ~RCC_APB1Periph_TIM2;

	//Set prescaler
	TIM2->PSC = 23; //divide by 24
	
	TIM2->ATRLR = CTR_RELOAD;

	TIM2->CHCTLR1 |= TIM_OC2M_2 | TIM_OC2M_1 | TIM_OC2PE;

	TIM2->CH2CVR = CTR_RELOAD/2;

	TIM2->CTLR1 |= TIM_ARPE;

	TIM2->CCER |= TIM_CC2E | TIM_CC2P;

	TIM2->SWEVGR |= TIM_UG; //initialize counter
}


#define LED PA1
#define UART_TX PD5
#define UART_RX PD6

int main(){
	SystemInit();

	funGpioInitAll(); // Enable GPIOs
	init_tone_timer();
	funPinMode(LED, FUN_OUTPUT);
	funPinMode(PC2, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP_AF);

	CMT2119A_settings_t tx_settings = {
		.freq_out_hz = 430100000,
		.fsk_dev_hz = 2000,
		.gfsk_rate_bps = 1600, //minimum required for 1khz tone in GFSK mode
		.power_output_dbm = -10,
		.modulation = MOD_FSK,
		.off_time = OFF_90MS,
		.pa_ramp_time = 0,
		.rising_edge_start = true,
		.invert_symbols = false,
		.xo_current_boost = false,
	};

	CMT2119A_init(PC1, PC2, &tx_settings); //use default frequency

	while(1)
	{
		funDigitalWrite(LED, FUN_LOW);
		//CMT2119A_update(); 
		//frequency can also be changed live, but transmitting the PLL parameters takes more time if we don't need to do so
		TWI_reset();
		Delay_Ms(2);
		CMT2119A_TWI_OFF();
		TWI_DAT_HIGH //wake up from sleep mode
		Delay_Us(1);
		TWI_DAT_LOW
		encode_morse(morse_string);
		TWI_reset(); //turns transmitter off
		//to transmit continuously, leave the DAT pin high
		funDigitalWrite(LED, FUN_HIGH);
		Delay_Ms(1000);
	}
}
