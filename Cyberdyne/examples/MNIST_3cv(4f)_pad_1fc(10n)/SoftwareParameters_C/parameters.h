#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <inttypes.h>

#define INPUT_CHANNELS	1
#define INPUT_LINES	28
#define INPUT_COLS	28

#define LAYERS	7

#define MAX_FILTERS	4

#define MAX_NEURONS	10

#define FC_OUT_SIZE	10

#define WEIGHT_SHIFT	7

#define SHARED_FMAP_SIZE	3600


#define RELU(x)	((x < 0) ? 0 : x)
#define APPLY_RELU	1
#define NO_RELU	0

#define CONV            1
#define MAX_POOL        2
#define FULL_CONNECTED  3

typedef int16_t fmap_dtype;

const int8_t filters_conv0[4][1][3][3] = {
	{
		{{-7, -6, 4}, {6, 6, 6}, {6, -5, -5}}
	},
	{
		{{5, 5, 6}, {-6, 6, 7}, {-7, -7, -8}}
	},
	{
		{{4, 4, 4}, {-5, -4, 6}, {-6, 5, 6}}
	},
	{
		{{7, -5, -6}, {6, -5, -6}, {5, 6, -4}}
	}
};
const int16_t bias_conv0[4] = {-7, -56, -10, -96};

const int8_t filters_conv1[4][4][3][3] = {
	{
		{{5, 5, 7}, {-4, 5, 5}, {-6, -6, 3}},
		{{4, 5, 5}, {6, 6, 6}, {6, 6, -4}},
		{{5, 4, 5}, {-3, -4, 5}, {-5, -6, 4}},
		{{5, -4, 3}, {0, -6, -4}, {-6, 3, -5}}
	},
	{
		{{5, 4, 5}, {3, -3, -5}, {-4, -5, 4}},
		{{-6, -6, 3}, {-7, -6, -4}, {-6, -4, 5}},
		{{6, 6, 5}, {5, -4, -5}, {-3, 5, -4}},
		{{6, 4, 0}, {4, -4, 6}, {6, 7, 6}}
	},
	{
		{{-5, -5, 2}, {-2, -5, 6}, {-6, -5, 5}},
		{{5, -3, -2}, {5, -2, -4}, {1, -5, -6}},
		{{-5, -5, 7}, {-6, -5, 7}, {-7, -5, 7}},
		{{5, -4, -4}, {2, -5, -5}, {-4, -6, 0}}
	},
	{
		{{6, -3, -6}, {6, 4, -6}, {3, -5, -6}},
		{{0, 4, -5}, {4, 3, -6}, {6, -4, -5}},
		{{7, 6, -5}, {6, -3, -7}, {-6, -7, -5}},
		{{5, -5, 5}, {0, -3, -3}, {-5, -5, -5}}
	}
};
const int16_t bias_conv1[4] = {-35, -16, -97, 2};

const int8_t filters_conv2[4][4][3][3] = {
	{
		{{2, -5, -6}, {6, -4, 5}, {4, -4, 6}},
		{{-3, -2, -7}, {5, 4, 7}, {3, 3, 6}},
		{{-7, -6, 5}, {-4, 5, -5}, {-6, 4, -4}},
		{{-5, 4, -6}, {-5, -4, -4}, {5, 7, -6}}
	},
	{
		{{3, 3, 4}, {-3, -5, -3}, {6, -5, -7}},
		{{4, -6, 6}, {-6, 6, 5}, {5, -5, -7}},
		{{-4, 6, 5}, {-6, 6, -3}, {-4, 5, -3}},
		{{-4, 6, 3}, {-5, 7, 7}, {6, 4, 5}}
	},
	{
		{{-4, 5, 8}, {-5, 3, -4}, {3, 5, 6}},
		{{5, -4, -5}, {-4, -7, -6}, {-2, 6, 5}},
		{{-6, 4, -6}, {-5, -6, 5}, {6, 5, -5}},
		{{3, 7, 6}, {-6, 5, 5}, {4, 5, -3}}
	},
	{
		{{5, -6, 3}, {-6, -7, -4}, {-6, -7, 3}},
		{{7, -4, -6}, {6, -5, 3}, {6, 6, 6}},
		{{-5, 5, 5}, {6, 5, 5}, {4, 5, -5}},
		{{-2, -3, -3}, {5, 6, 6}, {5, 2, 5}}
	}
};
const int16_t bias_conv2[4] = {78, -87, -28, 78};

const int8_t weights_fc0[10][36] = {
	{-4, 4, 6, 6, 4, 6, -4, 5, -6, -4, -6, -6, 4, 3, -4, -6, -4, 7, 0, 4, 5, -4, 5, -4, -7, -4, 5, -5, -4, -4, 6, 6, 4, -5, 5, -6},
	{-2, 6, -5, -7, -3, -6, 4, -2, 5, 4, 5, 5, -5, 6, 3, 4, -3, -7, -4, -4, 4, 5, -5, -6, 6, -3, -4, -5, 5, 7, -4, -3, 7, 5, 4, -3},
	{-5, -3, -3, 4, -3, -5, 7, 6, 5, 4, -5, -7, 6, 3, 5, 7, -3, -5, -4, -5, -5, 7, 6, -7, 5, -3, 6, -3, -5, -4, -4, 4, 7, -4, -5, -6},
	{-6, -3, 7, -6, -5, 5, -4, -5, 6, 4, -6, -3, 0, 5, 5, -6, 5, 6, -3, -6, -3, 7, -4, -4, -4, 7, -4, 0, -6, 6, -6, -5, 4, -5, -6, 6},
	{5, 4, 4, 6, -3, -6, -7, 5, -6, -4, 4, -5, -6, -6, 5, 6, 6, -6, 3, -3, 6, -5, -5, -5, 7, 6, -2, 4, 6, 7, -3, 4, -5, -6, 0, -4},
	{5, 2, 4, -3, 4, 3, 0, 2, 5, -3, 5, 6, -4, 5, -7, -3, 5, 5, -6, 6, 2, -5, 5, 7, 5, 6, -5, -5, 6, -5, -4, -5, -6, -5, 3, 6},
	{3, -4, -6, 6, 0, 7, -5, -3, 5, 5, 7, 7, -5, -3, -5, 5, -5, 7, -4, -6, 3, -7, 4, 7, -7, -4, 4, -3, 7, 0, 4, 7, -5, -5, -6, -6},
	{3, -5, -6, -6, 6, 6, -5, 5, -6, -2, -5, 5, 7, -4, 4, 5, 5, -5, 3, -5, 6, 6, 6, -7, 6, -4, 3, -2, -5, -4, -5, -4, 4, -4, 7, 6},
	{6, 5, 5, 6, 5, 4, 7, -6, 6, 3, 4, -2, -6, -4, 7, -6, -6, 5, 5, 4, -4, 0, -3, 4, -4, -6, 7, 6, -4, -4, -5, -5, -4, 6, -4, 6},
	{-3, 3, -4, 6, 4, 5, -6, 5, -6, 3, -6, 0, -6, -4, 6, -5, 5, -6, 5, 6, -7, -6, 5, -5, 6, 6, -7, 6, -5, -6, 6, -6, 6, 6, 3, 6}
};
const int16_t bias_fc0[10] = {196, -97, -128, -309, 5, 107, -114, 173, 71, 119};

struct Parameters {
    uint8_t type;
    
    /* Input feature map*/
    uint8_t fmap_in_lines;
    uint8_t fmap_in_cols;
    uint32_t fmap_in_channel_size;
    uint32_t fmap_in_channels;  /* Same as number of filter kernels */
    uint8_t pad;
    uint32_t padded_fmap;
    
    /* Output feature map*/
    uint8_t fmap_out_lines;
    uint8_t fmap_out_cols;
    uint32_t fmap_out_channel_size;
    uint32_t fmap_out_channels;
    uint32_t neurons;
    
    /* Filter features */
    const int8_t *filters;
    const int16_t *bias;
    uint8_t kernel_order;
    uint8_t kernel_size;
    uint8_t stride;
    uint16_t next_fmap_in_line;
    uint8_t apply_relu;    
};

struct Parameters parameters[LAYERS];

struct Parameters *GetLayersParameters() {

	/* Layer 0 */
	parameters[0].type = CONV;
	parameters[0].fmap_in_lines = 28;
	parameters[0].fmap_in_cols = 28;
	parameters[0].fmap_in_channels = 1;
	parameters[0].fmap_in_channel_size = 784;

	parameters[0].fmap_out_lines = 28;
	parameters[0].fmap_out_cols = 28;
	parameters[0].fmap_out_channels = 4;
	parameters[0].fmap_out_channel_size = 784;

	parameters[0].pad = 1;
	parameters[0].filters = (const int8_t *)filters_conv0;
	parameters[0].bias = (const int16_t *)bias_conv0;
	parameters[0].kernel_order = 3;
	parameters[0].kernel_size = 9;
	parameters[0].stride = 1;
	parameters[0].next_fmap_in_line = 28;
	parameters[0].apply_relu = 0;

	/* Layer 1 */
	parameters[1].type = MAX_POOL;
	parameters[1].fmap_in_lines = 28;
	parameters[1].fmap_in_cols = 28;
	parameters[1].fmap_in_channels = 4;
	parameters[1].fmap_in_channel_size = 784;

	parameters[1].fmap_out_lines = 14;
	parameters[1].fmap_out_cols = 14;
	parameters[1].fmap_out_channels = 4;
	parameters[1].fmap_out_channel_size = 196;


	/* Layer 2 */
	parameters[2].type = CONV;
	parameters[2].fmap_in_lines = 14;
	parameters[2].fmap_in_cols = 14;
	parameters[2].fmap_in_channels = 4;
	parameters[2].fmap_in_channel_size = 196;

	parameters[2].fmap_out_lines = 14;
	parameters[2].fmap_out_cols = 14;
	parameters[2].fmap_out_channels = 4;
	parameters[2].fmap_out_channel_size = 196;

	parameters[2].pad = 1;
	parameters[2].filters = (const int8_t *)filters_conv1;
	parameters[2].bias = (const int16_t *)bias_conv1;
	parameters[2].kernel_order = 3;
	parameters[2].kernel_size = 9;
	parameters[2].stride = 1;
	parameters[2].next_fmap_in_line = 14;
	parameters[2].apply_relu = 0;

	/* Layer 3 */
	parameters[3].type = MAX_POOL;
	parameters[3].fmap_in_lines = 14;
	parameters[3].fmap_in_cols = 14;
	parameters[3].fmap_in_channels = 4;
	parameters[3].fmap_in_channel_size = 196;

	parameters[3].fmap_out_lines = 7;
	parameters[3].fmap_out_cols = 7;
	parameters[3].fmap_out_channels = 4;
	parameters[3].fmap_out_channel_size = 49;


	/* Layer 4 */
	parameters[4].type = CONV;
	parameters[4].fmap_in_lines = 7;
	parameters[4].fmap_in_cols = 7;
	parameters[4].fmap_in_channels = 4;
	parameters[4].fmap_in_channel_size = 49;

	parameters[4].fmap_out_lines = 7;
	parameters[4].fmap_out_cols = 7;
	parameters[4].fmap_out_channels = 4;
	parameters[4].fmap_out_channel_size = 49;

	parameters[4].pad = 1;
	parameters[4].filters = (const int8_t *)filters_conv2;
	parameters[4].bias = (const int16_t *)bias_conv2;
	parameters[4].kernel_order = 3;
	parameters[4].kernel_size = 9;
	parameters[4].stride = 1;
	parameters[4].next_fmap_in_line = 7;
	parameters[4].apply_relu = 0;

	/* Layer 5 */
	parameters[5].type = MAX_POOL;
	parameters[5].fmap_in_lines = 7;
	parameters[5].fmap_in_cols = 7;
	parameters[5].fmap_in_channels = 4;
	parameters[5].fmap_in_channel_size = 49;

	parameters[5].fmap_out_lines = 3;
	parameters[5].fmap_out_cols = 3;
	parameters[5].fmap_out_channels = 4;
	parameters[5].fmap_out_channel_size = 9;


	/* Layer 6 */
	parameters[6].type = FULL_CONNECTED;
	parameters[6].fmap_in_lines = 3;
	parameters[6].fmap_in_cols = 3;
	parameters[6].fmap_in_channels = 4;
	parameters[6].fmap_in_channel_size = 9;

	parameters[6].filters = (const int8_t *)weights_fc0;
	parameters[6].bias = (const int16_t *)bias_fc0;
	parameters[6].neurons = 10;
	parameters[6].apply_relu = 0;

	return parameters;
}



#endif