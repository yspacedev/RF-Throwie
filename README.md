## About
The RF throwie is an ultra-cheap sub-GHz transmitter using the CMOSTek/HopeRF $1 CMT2119A RF transmitter and $0.10 CH32V003 32-bit RISC-V microcontroller on a 20x20mm 2-layer board.

Called a "throwie" because it's designed to be cheap enough (almost disposable) that you can throw it anywhere for a foxhunt or other RF application.

The CMT2119A supports (G)FSK/OOK modulation and a tuning range of 157MHz - 1297MHz (according to G4EML). While it's meant to be configured with CMOSTek's RFPDK and their proprietary software, it can be configured in-situ though a 2-wire interface that writes to registers on the chip. The registers have been partly reverse engineered in https://github.com/g4eml/RP2040_Synth, and this project aims to reverse engineer them more fully to allow for full customization without the RFPDK. The CMT2119A can also be used as a general-purpose frequency synthesizer, which opens up a lot of possibilities for cheap SDRs and custom RF frontends. 

### Firmware

The CH32V003 is programmed with cnlohr's ch32fun stack and uses bitbanging to generate the TWI signals. The current firmware transmits Morse code on the UHF ham radio band (70cm band) that can be picked up by an SDR using NBFM demodulation or a cheap handheld transciever like a Baofeng or Quansheng UV-K5. In theory more modulations are possible, including FSK-based modes. However, due to the long (370us) tuning time and coarse frequency resolution (200~400Hz) of the frequency synthesizer, phase modulation through frequency modulation is likely not possible with unmodified hardware. Maybe using dithering to feed the crystal input an adjustable average frequency will allow for finer frequency resolution as well as phase adjustment, but this has not been tested.

For development and compilation, you will need to set up CH32fun per the instructions on their github. If you use clangd or Microsoft Intellisense with VSCode/VSCodium, those options are already set up, but you will need to change the path for the libraries or use `compiledb make` to generate the `compile_commands.json` file.

### Board/hardware

The RF throwie board itself supports an input voltage of 3.7v - 18v, and has an onboard LED and the CH32V003 UART interface exposed. There are also 2 jumpers on the back that connect the UART pins to the CMT2119A TWI pins to turn it into a breakout board. For the RF chip, there's a up to 7th order LC filter and matching network feeding an SMA connector or simple wire antenna.

![Board Front](https://github.com/yspacedev/RF-Throwie/blob/master/resources/RF%20Throwie%20Front.png)
![Board Back](https://github.com/yspacedev/RF-Throwie/blob/master/resources/RF%20Throwie%20Back.png)
![Board Image](https://github.com/yspacedev/RF-Throwie/blob/master/resources/RF%20Throwie%20Board.webp)


## Tuning

The RF output pin has a frequency dependent impedance, and since CMOSTek only provided the impedance at 4 test frequencies, I used a Python curve fitter to get an empirical equation for the output impedance. This Desmos graph implements the formula: https://www.desmos.com/calculator/a3xyh0sjgi

To create a custom matching network for the transmitter, use https://home.sandiego.edu/~ekim/e194rfs01/jwmatcher/matcher2.html to create a match to 50 ohms and then use https://markimicrowave.com/technical-resources/tools/lc-filter-design-tool/ to generate a filter network.

## Extra Information

Extra information like LCSC part numbers for certain components is included in the KiCAD schematic.

It's designed to be fabricated on a 1.6mm FR4 2 layer board.
