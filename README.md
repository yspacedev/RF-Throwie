## About
The RF throwie is an ultra-cheap sub-GHz transmitter using the CMOSTek/HopeRF CMT2119A RF transmitter and CH32V003 32-bit RISC-V microcontroller. 

The CMT2119A supports (G)FSK/OOK modulation and a tuning range of 157MHz - 1297MHz (according to G4EML). While it's meant to be configured with CMOSTek's RFPDK and their proprietary software, it can be configured in-situ though a 2-wire interface that writes to registers on the chip. The registers have been partly reverse engineered in https://github.com/g4eml/RP2040_Synth, and this project aims to reverse engineer them more fully. The CMT2119A can also be used as a general-purpose frequency synthesizer, which opens up a lot of possibilities for cheap SDRs and custom RF frontends. It may also be able to do more advanced modulation through more direct control. Perhaps it can generate LoRa chirps (and thus LoRa) by adjusting the PLL live and/or abusing GFSK.

The CH32V003 is programmed with cnlohr's ch32fun stack and uses bitbanging to generate the TWI signals. (software currently in development since I don't have boards yet)

The RF throwie board itself supports an input voltage of 3.7v - 18v, and has an onboard LED and the CH32V003 UART interface exposed. For the RF chip, there's a up to 7th order LC filter and matching network feeding an SMA connector or simple wire antenna.

(Board pictures)

## Tuning

The RF output pin has a frequency dependent impedance, and since CMOSTek only provided the impedance at 4 test frequencies, I used a Python curve fitter to get an empirical equation for the output impedance. This Desmos graph implements the formula: https://www.desmos.com/calculator/a3xyh0sjgi

To create a custom matching network for the transmitter, use https://home.sandiego.edu/~ekim/e194rfs01/jwmatcher/matcher2.html to create a match to 50 ohms and then use https://markimicrowave.com/technical-resources/tools/lc-filter-design-tool/ to generate a filter network.

Note: this tuning method is not yet verified since I don't have boards yet

## Extra Information

Extra information like LCSC part numbers for certain components is included in the KiCAD schematic.

It's designed to be fabricated on a 1.6mm FR4 2 layer board.
