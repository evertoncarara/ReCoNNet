library IEEE;
use ieee.numeric_std.all;
use IEEE.std_logic_1164.all;
use std.textio.all;

package SkyNet_pkg is

    constant FREQ_BAUD_RATE     : integer := 868; -- 115200 baud rate at 100MHz

    constant CONV_LAYERS        : integer := 6; -- Number of convolution layers
    type conv_layers_parameters_t is array (0 to CONV_LAYERS - 1) of integer;
    constant STRIDE             : conv_layers_parameters_t := (1, 1, 1, 1, 1, 1);
    constant KERNEL_ORDER       : conv_layers_parameters_t := (3, 3, 3, 3, 3, 3);
    constant KERNEL_LENGTH      : conv_layers_parameters_t := (9, 9, 9, 9, 9, 9);
    constant CONV_INPUT_WIDTH   : conv_layers_parameters_t := (32, 32, 16, 16, 8, 8);
    constant CONV_INPUT_HEIGHT  : conv_layers_parameters_t := (32, 32, 16, 16, 8, 8);
    constant FMAP_LINES         : conv_layers_parameters_t := (32, 16, 16, 8, 8, 4);
    constant FMAP_COLUMNS       : conv_layers_parameters_t := (32, 16, 16, 8, 8, 4);
    constant FMAP_MAX_LINES     : integer := 32;
    constant FMAP_MAX_COLUMNS   : integer := 32;
    constant POOL_KERNEL_ORDER  : conv_layers_parameters_t := (1, 2, 1, 2, 1, 2);
    constant PAD                : conv_layers_parameters_t := (1, 1, 1, 1, 1, 1);
    constant FMAP_WR_ADDR_INIT  : conv_layers_parameters_t := (85, 52, 35, 18, 9, 0);  -- Parameter used only when PAD > 0 
    constant FMAPS_ADDR_WIDTH   : integer := 11;  -- Feature map memory address bus width
    -- NEW_LINE includes convolutional and MaxPool2d layers stride
    -- Supports only square strides
    constant NEW_LINE           : conv_layers_parameters_t := (32, 64, 16, 32, 8, 16); -- Offset used to set the memory address of a new feature map line
    constant CHANNELS           : conv_layers_parameters_t := (1, 36, 36, 36, 36, 36);  -- Number of input channels in each convolution layer
    constant FILTER_START_ADDR  : conv_layers_parameters_t := (0, 3, 45, 87, 129, 171);  -- The starting memory address of filters for each layer
    constant CONV_LAYER_FILTERS : conv_layers_parameters_t := (36, 36, 36, 36, 36, 36);
    constant FILTERS_ADDR_WIDTH : integer := 8;  -- Address bus width of convolution layer filter/sign memories
    constant MAX_FILTERS        : integer := 36;  -- Number of filters in the convolution layer with more filters

    constant FULL_LAYERS        : integer := 1;  -- Number of full connected layers
    type fc_layers_parameters_t is array (0 to FULL_LAYERS - 1) of integer;
    constant NEURONS            : fc_layers_parameters_t := (10, others=>0);
    constant MAX_NEURONS        : integer := 10;  -- Number of neurons in the full connected layer with more neurons
    constant NEURON_INPUTS_PER_CHANNEL: integer := 16;  -- Number of input features in each neuron
    constant FULL_LAYER_ADDR_WIDTH  : integer := 10;  -- Address bus width of full connected layer filter/sign memories
    constant WEIGHT_SHIFT       : integer := 7;
    constant BIAS_SHIFT         : integer := 7;
    constant MORE_WEIGHTS       : integer := 576;
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
        "raw/FilterWeights_003.txt",
        "raw/FilterWeights_004.txt",
        "raw/FilterWeights_005.txt",
        "raw/FilterWeights_006.txt",
        "raw/FilterWeights_007.txt",
        "raw/FilterWeights_008.txt",
        "raw/FilterWeights_009.txt",
        "raw/FilterWeights_010.txt",
        "raw/FilterWeights_011.txt",
        "raw/FilterWeights_012.txt",
        "raw/FilterWeights_013.txt",
        "raw/FilterWeights_014.txt",
        "raw/FilterWeights_015.txt",
        "raw/FilterWeights_016.txt",
        "raw/FilterWeights_017.txt",
        "raw/FilterWeights_018.txt",
        "raw/FilterWeights_019.txt",
        "raw/FilterWeights_020.txt",
        "raw/FilterWeights_021.txt",
        "raw/FilterWeights_022.txt",
        "raw/FilterWeights_023.txt",
        "raw/FilterWeights_024.txt",
        "raw/FilterWeights_025.txt",
        "raw/FilterWeights_026.txt",
        "raw/FilterWeights_027.txt",
        "raw/FilterWeights_028.txt",
        "raw/FilterWeights_029.txt",
        "raw/FilterWeights_030.txt",
        "raw/FilterWeights_031.txt",
        "raw/FilterWeights_032.txt",
        "raw/FilterWeights_033.txt",
        "raw/FilterWeights_034.txt",
        "raw/FilterWeights_035.txt");

    type FilterSignsImageFileName_t is array (0 to MAX_FILTERS - 1) of string(1 to 23);
    constant FILTER_SIGNS_FILES : FilterSignsImageFileName_t := (
        "raw/FilterSigns_000.txt",
        "raw/FilterSigns_001.txt",
        "raw/FilterSigns_002.txt",
        "raw/FilterSigns_003.txt",
        "raw/FilterSigns_004.txt",
        "raw/FilterSigns_005.txt",
        "raw/FilterSigns_006.txt",
        "raw/FilterSigns_007.txt",
        "raw/FilterSigns_008.txt",
        "raw/FilterSigns_009.txt",
        "raw/FilterSigns_010.txt",
        "raw/FilterSigns_011.txt",
        "raw/FilterSigns_012.txt",
        "raw/FilterSigns_013.txt",
        "raw/FilterSigns_014.txt",
        "raw/FilterSigns_015.txt",
        "raw/FilterSigns_016.txt",
        "raw/FilterSigns_017.txt",
        "raw/FilterSigns_018.txt",
        "raw/FilterSigns_019.txt",
        "raw/FilterSigns_020.txt",
        "raw/FilterSigns_021.txt",
        "raw/FilterSigns_022.txt",
        "raw/FilterSigns_023.txt",
        "raw/FilterSigns_024.txt",
        "raw/FilterSigns_025.txt",
        "raw/FilterSigns_026.txt",
        "raw/FilterSigns_027.txt",
        "raw/FilterSigns_028.txt",
        "raw/FilterSigns_029.txt",
        "raw/FilterSigns_030.txt",
        "raw/FilterSigns_031.txt",
        "raw/FilterSigns_032.txt",
        "raw/FilterSigns_033.txt",
        "raw/FilterSigns_034.txt",
        "raw/FilterSigns_035.txt");

end SkyNet_pkg;