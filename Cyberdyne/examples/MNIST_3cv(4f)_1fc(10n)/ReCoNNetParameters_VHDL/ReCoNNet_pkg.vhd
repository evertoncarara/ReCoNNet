library IEEE;
use ieee.numeric_std.all;
use IEEE.std_logic_1164.all;
use std.textio.all;

package ReCoNNet_pkg is

    constant FREQ_BAUD_RATE     : integer := 868; -- 115200 baud rate at 100MHz

    constant FMAP_MAX_LINES     : integer := 13;
    constant FMAP_MAX_COLUMNS   : integer := 13;
    constant FMAPS_ADDR_WIDTH   : integer := 8;  -- Feature map memory address bus width
    -- NEW_LINE includes convolutional and MaxPool2d layers stride
    -- Supports only square strides
    constant FILTERS_ADDR_WIDTH : integer := 4;  -- Address bus width of convolution layer filter/sign memories
    constant MAX_FILTERS        : integer := 4;  -- Number of filters in the convolution layer with more filters
    constant FULL_LAYER_ADDR_WIDTH  : integer := 6;  -- Address bus width of full connected layer filter/sign memories
    constant CONFIG_ADDR_WIDTH  : integer := 6;
    constant CONV_ACC_WIDTH     : integer := 20; -- Convolution layer accumulators width
    constant NEURON_ACC_WIDTH   : integer := 20; -- Full connected layer accumulators width
    constant WEIGHT_WIDTH       : integer := 4;
    constant DATAPATH_WIDTH     : integer := 32;
    constant WEIGHTS_PER_LINE   : integer := 8; -- Number of weigths stored in a memory word (line)
    constant WEIGHTS_MEM_DATA_WIDTH   : integer := WEIGHT_WIDTH * WEIGHTS_PER_LINE;
    constant NEURONS_ADDR_WIDTH : integer := 4; -- Address bus width of NEURONS memory 

    -- types used in datapath
    type fmap_data_array_t  is array (0 to MAX_FILTERS - 1) of std_logic_vector(CONV_ACC_WIDTH - 1 downto 0);
    type conv_acc_array_t   is array (0 to MAX_FILTERS - 1) of SIGNED(CONV_ACC_WIDTH - 1 downto 0);

    -- multiplications
    type conv_mult_array_t      is array (0 to MAX_FILTERS - 1) of SIGNED(DATAPATH_WIDTH - 1 downto 0);

    -- weight memory addresses and weight selectors
    type filter_weights_line_array_t    is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHTS_MEM_DATA_WIDTH - 1 downto 0);
    type filter_signs_line_array_t      is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHTS_PER_LINE - 1 downto 0);
    type filter_weight_array_t          is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHT_WIDTH - 1 downto 0);
    type filter_sign_array_t            is array (0 to MAX_FILTERS - 1) of std_logic;

    type FilterWeightsImageFileName_t is array (0 to MAX_FILTERS - 1) of string(1 to 25);
    constant FILTER_WEIGTHS_FILES : FilterWeightsImageFileName_t := (
        "raw/FilterWeights_000.txt",
        "raw/FilterWeights_001.txt",
        "raw/FilterWeights_002.txt",
        "raw/FilterWeights_003.txt");

    type FilterSignsImageFileName_t is array (0 to MAX_FILTERS - 1) of string(1 to 23);
    constant FILTER_SIGNS_FILES : FilterSignsImageFileName_t := (
        "raw/FilterSigns_000.txt",
        "raw/FilterSigns_001.txt",
        "raw/FilterSigns_002.txt",
        "raw/FilterSigns_003.txt");

end ReCoNNet_pkg;