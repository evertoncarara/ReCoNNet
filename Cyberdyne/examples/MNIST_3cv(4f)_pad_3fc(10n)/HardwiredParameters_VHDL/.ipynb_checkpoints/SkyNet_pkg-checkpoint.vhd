library IEEE;
use ieee.numeric_std.all;
use IEEE.std_logic_1164.all;
use std.textio.all;

package SkyNet_pkg is

    constant FREQ_BAUD_RATE     : integer := 868; -- 115200 baud rate at 100MHz

    constant CONV_LAYERS        : integer := 3; -- Number of convolution layers
    type conv_layers_parameters_t is array (0 to CONV_LAYERS - 1) of integer;
    constant STRIDE             : conv_layers_parameters_t := (1, 1, 1);
    constant KERNEL_ORDER       : conv_layers_parameters_t := (3, 3, 3);
    constant KERNEL_LENGTH      : conv_layers_parameters_t := (9, 9, 9);
    constant CONV_INPUT_WIDTH   : conv_layers_parameters_t := (28, 14, 7);
    constant CONV_INPUT_HEIGHT  : conv_layers_parameters_t := (28, 14, 7);
    constant FMAP_LINES         : conv_layers_parameters_t := (14, 7, 3);
    constant FMAP_COLUMNS       : conv_layers_parameters_t := (14, 7, 3);
    constant FMAP_MAX_LINES     : integer := 14;
    constant FMAP_MAX_COLUMNS   : integer := 14;
    constant POOL_KERNEL_ORDER  : conv_layers_parameters_t := (2, 2, 2);
    constant PAD                : conv_layers_parameters_t := (1, 1, 1);
    constant FMAP_WR_ADDR_INIT  : conv_layers_parameters_t := (23, 8, 0);  -- Parameter used only when PAD > 0 
    constant FMAPS_ADDR_WIDTH   : integer := 8;  -- Feature map memory address bus width
    -- NEW_LINE includes convolutional and MaxPool2d layers stride
    -- Supports only square strides
    constant NEW_LINE           : conv_layers_parameters_t := (56, 28, 14); -- Offset used to set the memory address of a new feature map line
    constant CHANNELS           : conv_layers_parameters_t := (1, 4, 4);  -- Number of input channels in each convolution layer
    constant FILTER_START_ADDR  : conv_layers_parameters_t := (0, 3, 9);  -- The starting memory address of filters for each layer
    constant CONV_LAYER_FILTERS : conv_layers_parameters_t := (4, 4, 4);
    constant FILTERS_ADDR_WIDTH : integer := 4;  -- Address bus width of convolution layer filter/sign memories
    constant MAX_FILTERS        : integer := 4;  -- Number of filters in the convolution layer with more filters

    constant FULL_LAYERS        : integer := 3;  -- Number of full connected layers
    type fc_layers_parameters_t is array (0 to FULL_LAYERS - 1) of integer;
    constant NEURONS            : fc_layers_parameters_t := (20, 15, 10);  -- Number of neurons in each full connected layer
    constant MAX_NEURONS        : integer := 20;  -- Number of neurons in the full connected layer with more neurons
    constant NEURON_INPUTS_PER_CHANNEL: integer := 9;  -- Number of input features in each neuron
    constant FULL_LAYER_ADDR_WIDTH  : integer := 8;  -- Address bus width of full connected layer filter/sign memories
    constant WEIGHT_SHIFT       : integer := 7;
    constant BIAS_SHIFT         : integer := 7;
    constant MORE_WEIGHTS       : integer := 36;
    constant CONV_ACC_WIDTH     : integer := 20; -- Convolution layer accumulators width
    constant NEURON_ACC_WIDTH   : integer := 20; -- Full connected layer accumulators width
    constant WEIGHT_WIDTH       : integer := 4;
    constant DATAPATH_WIDTH     : integer := 32;
    constant WEIGHTS_PER_LINE   : integer := 8; -- Number of weigths stored in a memory word (line)
    constant WEIGHTS_MEM_DATA_WIDTH   : integer := WEIGHT_WIDTH * WEIGHTS_PER_LINE;
    constant NEURONS_ADDR_WIDTH : integer := 6; -- Address bus width of NEURONS memory

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

end SkyNet_pkg;