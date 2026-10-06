-- Limitations
--  - Layers: Convolutional, ReLU, MaxPool2d and FullConnected
--  - ReLU only after convolutional layers
--  - When using MaxPool2d, supports at most one after each convolutional layer
--  - Suport only square strides (convolutional/MaxPool2d)
--  - MaxPool2d stride is equal to MaxPool2d kernel order
--  - Padding: suports only the same pad (0 or 1) for width and height
--          * NEEDS MORE TEST 
--  - Full connected layers supports only ReLU as activation function

library IEEE;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.SkyNet_pkg.all;

entity SkyNet_serialFC is
    generic (
        SIGNED_DATA_I   : boolean := false;
        DATA_WIDTH      : integer := 8;
        ADDR_WIDTH      : integer := 8
    );
    port ( 
        clk             : in  std_logic;
        rst             : in  std_logic; 
        start_i         : in  std_logic;
        
        done_o          : out std_logic;
        load_input_led  : out std_logic;
		class_o			: out std_logic_vector(7 downto 0);
        
        -- Memory interface to input feature map
        data_i          : in  std_logic_vector(DATA_WIDTH - 1 downto 0); -- CIFAR10 shift = 7
        address_o       : out std_logic_vector(ADDR_WIDTH - 1 downto 0)
    );
end SkyNet_serialFC;

--      KERNELS vs FILTERS
-- A “Kernel” refers to a 2D array of weights. The term “filter” is for 3D structures of 
-- multiple kernels stacked together. For a 2D filter, filter is same as kernel. But for a 
-- 3D filter and most convolutions in deep learning, a filter is a collection of kernels. 
-- Each kernel is unique, emphasizing different aspects of the input channel.
-- https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215

architecture Behavioral of SkyNet_serialFC is

    type state_t       is (INIT, LOAD_INPUT, FINISH_MAC, FC_ADD_BIAS, CONV_ADD_BIAS, CONV_MEM_WRITE, RECEPTIVE_FILED_CONV, FULL_CONNECTED, FC_MEM_WRITE, START_NEW_FC_LAYER, READ_FIRST_WEIGHT, DONE);
    signal currentState_s, currentState_delayed_s       : state_t;
    
    signal k_ref_index_s        : integer;    -- beggining index of current kernel 
    signal k_col_index_s        : integer range 0 to 7; -- 0 TO ORDEM DO MAIOR FILTRO     -- "column" index of current kernel
    signal k_lin_index_s        : integer range 0 to 7; -- 0 TO ORDEM DO MAIOR FILTRO    -- "line" index of current kernel
    signal k_line_jump_s        : integer; -- TEM QUE O ENDEREÇO INICIAL DA ÚLTIMA LINHA DA IMAGEM DE ENTRADA necessary jump to reach next line of kernel
    
    signal channel_index_s      : integer range 0 to MAX_FILTERS - 1;-- input channel index
    
    signal fmap_col_index_s     : integer range 0 to FMAP_MAX_COLUMNS;  -- Current output feature map column
    signal fmap_lin_index_s     : integer range 0 to FMAP_MAX_LINES;    -- Current output feature map line
    
    signal weight_flat_index_s  : integer range 0 to MORE_WEIGHTS - 1; -- 0 TO MAX(ORDEM DO MAIOR FILTRO)^2, PESOS DE UM NEURONIO)       -- weight selector in kernel

    signal filter_weight_line_addr_s    : UNSIGNED(FILTERS_ADDR_WIDTH - 1 downto 0);     -- address to read weight memory
    signal weight_sel_s                 : integer range 0 to WEIGHTS_PER_LINE - 1;      -- selects one weight from line read from memory
    signal full_layer_weights_addr_s     : UNSIGNED(FULL_LAYER_ADDR_WIDTH - 1 downto 0);

    
    -- indexes relevant to layers
    signal conv_layer_index_s   : integer range 0 to CONV_LAYERS - 1;  -- layer/parameters index
    signal previous_layer_index_s, neuron_acc_index_delayed_s       : integer range 0 to MAX_NEURONS - 1;
    signal fc_layer_index_s     : integer range 0 to FULL_LAYERS - 1;  -- layer/parameters index

    -- memories signals
    signal flat_rd_addr_s       : integer;    --  index to access flattened image in memory

    
    signal fmap_wr_addr_s, fmap_wr_addr_delayed_s       : UNSIGNED(FMAPS_ADDR_WIDTH - 1 downto 0);
    signal fmap_rd_addr_s       : SIGNED(FMAPS_ADDR_WIDTH - 1 downto 0);
    signal fmap_i_s             : fmap_data_array_t;
    signal fmap_o_s             : fmap_data_array_t;
    signal fmap_wr_s            : std_logic;

    signal filter_weights_line_s, filter_weights_line_delayed_s: filter_weights_line_array_t;
    signal filter_signs_o_s     : filter_signs_line_array_t;
    signal neuron_weights_line_s, neuron_weights_line_delayed_s: std_logic_vector(WEIGHTS_MEM_DATA_WIDTH - 1 downto 0);
	signal neuron_signs_line_s 	: std_logic_vector(WEIGHTS_PER_LINE - 1 downto 0);

    -- weights and signs after selection
    signal filter_weight_s      : filter_weight_array_t;
	type filter_weight_diff_array_t is array (0 to MAX_FILTERS - 1) of std_logic_vector(WEIGHT_WIDTH downto 0);
    signal filter_weight_diff_s : filter_weight_diff_array_t;
    signal filter_sign_s, filter_sign_delyed_s        : filter_sign_array_t;
    signal neuron_weight_s      : std_logic_vector(WEIGHT_WIDTH - 1 downto 0); 
    signal neuron_weight_diff_s : std_logic_vector(WEIGHT_WIDTH downto 0);
    signal neuron_sign_s, neuron_sign_delayed_s        : std_logic; 

    -- layers inputs, outputs and control
    signal neuron_input_s       : SIGNED(DATAPATH_WIDTH - 1 downto 0);

    --  Multiplications
    signal conv_shift_s, conv_shift_delayed_s, conv_mult_s : conv_mult_array_t;
    signal neuron_shift_s, neuron_shift_delayed_s,  neuron_mult_s: SIGNED(DATAPATH_WIDTH - 1 downto 0);--neuron_mult_array_t;

    -- filters internal signals
    signal feature_s    : SIGNED(DATAPATH_WIDTH - 1 downto 0);
    
    -- Accumulators
    signal conv_acc_s   : conv_acc_array_t;

    
    -- MAX POOL
    signal pool_lin_s, pool_col_s : integer;
    signal end_of_pool: boolean;
    signal max_pool_s: conv_acc_array_t;
    signal pool_start_s: integer;
    
    signal line_start_s: integer;
	signal neuron_index_s: integer := 0;
	signal previous_neurons         : integer range 0 to 255;

        
    -- PAD
    signal virtual_col_s        : integer range -PAD(0) to CONV_INPUT_WIDTH(0) - 1 + PAD(0);
    signal virtual_lin_s        : integer range -PAD(0) to CONV_INPUT_HEIGHT(0) - 1 + PAD(0);
    signal pad_area, pad_data_s : boolean;
    
	-- Neuron memory signals
	signal neurons_wr_addr_s, neurons_rd_addr_s, neurons_wr_addr_delayed_s: UNSIGNED(NEURONS_ADDR_WIDTH - 1 downto 0);
	signal neurons_wr_s: std_logic;
	signal neurons_o_s: std_logic_vector(NEURON_ACC_WIDTH - 1 downto 0);
	signal neuron_acc_s, neuron_max_s : SIGNED(NEURON_ACC_WIDTH - 1 downto 0);
	signal class_s: integer range 0 to 255;
	
begin

	class_o <= STD_LOGIC_VECTOR(TO_UNSIGNED(class_s, class_o'length));

    load_input_led <= '1' when currentState_s = LOAD_INPUT else '0';
    
    address_o <= STD_LOGIC_VECTOR(TO_SIGNED(flat_rd_addr_s, address_o'length)); 
        
    pad_area <= true when virtual_col_s < 0 or virtual_lin_s < 0 or 
                          virtual_col_s > CONV_INPUT_WIDTH(conv_layer_index_s) - 1 or
                          virtual_lin_s > CONV_INPUT_HEIGHT(conv_layer_index_s) - 1 else
                false;
    
   --################
   --#              #
   --# CONTROL ZONE #
   --#              #
   --################

    STATE_MACHINE: process(clk, rst)
        variable new_k_ref_index_var, new_virtual_col_var, new_virtual_lin_var : integer;
    begin
    
        if rst = '1' then
         
            currentState_s <= INIT;
               
        elsif rising_edge(clk) then
			
			-- Delayed signals (1 cycle) used to support 2 stages MAC
			currentState_delayed_s <= currentState_s;
			fmap_wr_addr_delayed_s <= fmap_wr_addr_s;
			neurons_wr_addr_delayed_s <= neurons_wr_addr_s;
			neuron_acc_index_delayed_s <= neuron_index_s;
        
            -- Signalize the feature_s is coming from pad area
            -- 1 cycle delay due to memory read
            pad_data_s <= pad_area;

            CASE currentState_s is
                when INIT =>
                    conv_layer_index_s  <= 0;
                    fc_layer_index_s    <= 0;
                    channel_index_s     <= 0;
                    k_lin_index_s       <= 0;
                    k_col_index_s       <= 0;
                    fmap_col_index_s    <= 0;
                    fmap_lin_index_s    <= 0;
                    weight_flat_index_s <= 0;
                    weight_sel_s        <= 0;
                    filter_weight_line_addr_s <= TO_UNSIGNED(FILTER_START_ADDR(0), filter_weight_line_addr_s'length);
                    full_layer_weights_addr_s <= (others=>'0');
					neurons_rd_addr_s <= (others=>'0');
                    
                    if PAD(0) = 1 then
                        k_ref_index_s   <= -(CONV_INPUT_WIDTH(0) + PAD(0));
                        flat_rd_addr_s  <= -(CONV_INPUT_WIDTH(0) + PAD(0));
                        pool_start_s    <= -(CONV_INPUT_WIDTH(0) + PAD(0));                     
                        line_start_s    <= -(CONV_INPUT_WIDTH(0) + PAD(0));
                    else -- No padding at first convolutional layer
                        k_ref_index_s   <= 0;
                        flat_rd_addr_s  <= 0;
                        pool_start_s    <= 0;
                        line_start_s    <= 0;
                    end if;
                     
                    fmap_wr_addr_s  <= TO_UNSIGNED(FMAP_WR_ADDR_INIT(0), fmap_wr_addr_s'length);
                     
                    virtual_col_s   <= -PAD(0); 
                    virtual_lin_s   <= -PAD(0);
                    k_line_jump_s   <= CONV_INPUT_WIDTH(0); 
                    
                    currentState_s  <= LOAD_INPUT;
                    
                    pool_lin_s <= 0;
                    pool_col_s <= 0;
                    end_of_pool <= false;
                    
                    previous_layer_index_s <= 0;

                when LOAD_INPUT => 
              
                    if start_i = '1' then
                        currentState_s <= READ_FIRST_WEIGHT;
                    else
                        currentState_s <= LOAD_INPUT;
                    end if;

               
                when READ_FIRST_WEIGHT =>

                    flat_rd_addr_s <= flat_rd_addr_s + 1;
                    virtual_col_s <= virtual_col_s + 1;
                    k_col_index_s  <= k_col_index_s + 1;
                    currentState_s <= RECEPTIVE_FILED_CONV;

                when FINISH_MAC =>

                    weight_flat_index_s <= 0;
                    weight_sel_s        <= 0;
                    channel_index_s     <= 0;

                    currentState_s <= CONV_ADD_BIAS;

                    -- Verifies if the feature map is complete
                    if fmap_lin_index_s = FMAP_LINES(conv_layer_index_s) then

                        if conv_layer_index_s < (CONV_LAYERS - 1) then
                            filter_weight_line_addr_s <= TO_UNSIGNED(FILTER_START_ADDR(conv_layer_index_s + 1), filter_weight_line_addr_s'length);
                            flat_rd_addr_s <= FMAP_WR_ADDR_INIT(conv_layer_index_s + 1);
                            virtual_lin_s <= -PAD(conv_layer_index_s + 1);
                        else
                            flat_rd_addr_s <= 0; -- Ready to compute Full Connected layer(s)
                            weight_sel_s <= 0;
                        end if;
            
                    elsif conv_layer_index_s < CONV_LAYERS then
                        filter_weight_line_addr_s <= TO_UNSIGNED(FILTER_START_ADDR(conv_layer_index_s), filter_weight_line_addr_s'length);
                    end if;
                    
                when CONV_ADD_BIAS =>                    
                    if end_of_pool then
                        end_of_pool <= false;
                        currentState_s <= CONV_MEM_WRITE;
                    else
                        k_col_index_s  <= k_col_index_s + 1;
                        flat_rd_addr_s <= flat_rd_addr_s + 1;
                        virtual_col_s <= virtual_col_s + 1;
                        currentState_s <= RECEPTIVE_FILED_CONV;
                    end if;


                -- write on memory and sometimes change layer
                when CONV_MEM_WRITE =>
                    
                    flat_rd_addr_s <= flat_rd_addr_s + 1;
                    virtual_col_s <= virtual_col_s + 1; 
                    
                    -- if layer field count is reached, go to next layer
                    if fmap_lin_index_s = FMAP_LINES(conv_layer_index_s) then
                        fmap_lin_index_s <= 0;
                        fmap_col_index_s <= 0;
                        channel_index_s <= 0;
               
                        if conv_layer_index_s < CONV_LAYERS  - 1 then
                            conv_layer_index_s <= conv_layer_index_s + 1;
                            k_ref_index_s <= FMAP_WR_ADDR_INIT(conv_layer_index_s + 1);
                            pool_start_s <= FMAP_WR_ADDR_INIT(conv_layer_index_s + 1); 
                            line_start_s <= FMAP_WR_ADDR_INIT(conv_layer_index_s + 1);
                            k_col_index_s  <= 1; 
                            fmap_wr_addr_s <= TO_UNSIGNED(FMAP_WR_ADDR_INIT(conv_layer_index_s + 1), fmap_wr_addr_s'length);
                            k_line_jump_s  <= CONV_INPUT_WIDTH(conv_layer_index_s + 1);
                            currentState_s <= RECEPTIVE_FILED_CONV;
                        else
                            currentState_s <= FULL_CONNECTED;
                        end if;
               
                    else 
                        -- change write adress for next memory write      
                        fmap_wr_addr_s <= fmap_wr_addr_s + 1;

                        k_col_index_s  <= 1;                
                        currentState_s <= RECEPTIVE_FILED_CONV;
               
                    end if;

 
                -- convolution inference
                when RECEPTIVE_FILED_CONV =>
                      
                    -- Verifies if all kernel weights were read (flat kernel view)
                    if weight_flat_index_s = KERNEL_LENGTH(conv_layer_index_s) - 1 then
                        weight_flat_index_s <= 0;
                        
                        -- Next channel
                        if channel_index_s < CHANNELS(conv_layer_index_s) - 1 then
                            channel_index_s <= channel_index_s + 1;
                        end if;                
                    else
                        weight_flat_index_s <= weight_flat_index_s + 1;
                    end if;
                            
                    -- Verifies if all weights in a memory line were read
                    if weight_sel_s = WEIGHTS_PER_LINE - 1 then 
                        weight_sel_s <= 0;
                    else
                        weight_sel_s <= weight_sel_s + 1;
                    end if;
                    
                    -- Verifies if all CONV_LAYER_FILTERS were read (updates weight memory address)
                    if weight_sel_s = WEIGHTS_PER_LINE - 2 then 
                        filter_weight_line_addr_s <= filter_weight_line_addr_s + 1;
                    end if;
                    
                    -- Updates the state/feature map memory flat address
                    -- jump to next column of kernel
                    if (k_col_index_s < KERNEL_ORDER(conv_layer_index_s) - 1) then
                        
                        k_col_index_s  <= k_col_index_s + 1;
                        flat_rd_addr_s <= flat_rd_addr_s + 1;
                        virtual_col_s <= virtual_col_s + 1;
                        currentState_s <= RECEPTIVE_FILED_CONV;
                        
                    -- jump to next line of kernel                   
                    elsif (k_lin_index_s < KERNEL_ORDER(conv_layer_index_s) - 1) then
                    
                        k_lin_index_s  <= k_lin_index_s + 1;
                        k_col_index_s  <= 0;
                        flat_rd_addr_s <= k_ref_index_s + k_line_jump_s;-- + (PAD(0) * 2); 
                        virtual_col_s <= virtual_col_s - (KERNEL_ORDER(conv_layer_index_s) - 1);
                        virtual_lin_s <= virtual_lin_s + 1; 
          
                        -- preprare for next kernel line jump
                        k_line_jump_s <= k_line_jump_s + CONV_INPUT_WIDTH(conv_layer_index_s);
                        currentState_s <= RECEPTIVE_FILED_CONV;

                    -- jump to next channel of kernel
                    elsif channel_index_s < CHANNELS(conv_layer_index_s) - 1 then

                        k_col_index_s <= 0;
                        k_lin_index_s <= 0;
                        flat_rd_addr_s <= k_ref_index_s;
                        virtual_col_s <= virtual_col_s - (KERNEL_ORDER(conv_layer_index_s) - 1);
                        virtual_lin_s <= virtual_lin_s - (KERNEL_ORDER(conv_layer_index_s) - 1);

                        -- reset kernel line jump
                        k_line_jump_s  <= CONV_INPUT_WIDTH(conv_layer_index_s);
                        currentState_s <= RECEPTIVE_FILED_CONV;
                        
                    -- jump to next kernel of line
                    else
                        -- Feature map point generated
                        --  ENTRA AQUI QUANDO TERMINNA UM PONTO DO FEATURE MAP
                        -- Atualiza a posição do kernel (k_ref_index_s)
                        if pool_col_s < POOL_KERNEL_ORDER(conv_layer_index_s) - 1 then
                            pool_col_s <= pool_col_s + 1;
                            new_k_ref_index_var := k_ref_index_s + STRIDE(conv_layer_index_s);
                            k_ref_index_s <= new_k_ref_index_var;
                            new_virtual_col_var := virtual_col_s - (KERNEL_ORDER(conv_layer_index_s) - 1) + STRIDE(conv_layer_index_s); 
                            new_virtual_lin_var := virtual_lin_s - (KERNEL_ORDER(conv_layer_index_s) - 1); 
                            
                        elsif pool_lin_s < POOL_KERNEL_ORDER(conv_layer_index_s) - 1 then
                            pool_lin_s <= pool_lin_s + 1;
                            pool_col_s <= 0;
                            new_k_ref_index_var := k_ref_index_s - (POOL_KERNEL_ORDER(conv_layer_index_s) - 1) + CONV_INPUT_WIDTH(conv_layer_index_s);
                            k_ref_index_s <= new_k_ref_index_var;
                            new_virtual_col_var := virtual_col_s - (KERNEL_ORDER(conv_layer_index_s) - 1) - STRIDE(conv_layer_index_s); 
                            new_virtual_lin_var := virtual_lin_s - (KERNEL_ORDER(conv_layer_index_s) - 1) + STRIDE(conv_layer_index_s); 
                            
                        else -- End of pool (write max to mem) 
                            end_of_pool <= true;
                            pool_col_s <= 0;
                            pool_lin_s <= 0;
                            new_virtual_col_var := virtual_col_s - (KERNEL_ORDER(conv_layer_index_s) - 1) + STRIDE(conv_layer_index_s); 

                            if fmap_col_index_s < FMAP_COLUMNS(conv_layer_index_s) - 1 then
                                fmap_col_index_s <= fmap_col_index_s + 1;
                                
                                if POOL_KERNEL_ORDER(conv_layer_index_s) > 1 then
                                    new_k_ref_index_var := pool_start_s + POOL_KERNEL_ORDER(conv_layer_index_s);
                                    pool_start_s <= new_k_ref_index_var;
                                    new_virtual_lin_var := virtual_lin_s - (KERNEL_ORDER(conv_layer_index_s) - 1) - STRIDE(conv_layer_index_s); -- carara

                                else -- No MaxPool layer
                                    new_k_ref_index_var := k_ref_index_s + STRIDE(conv_layer_index_s);
                                    new_virtual_lin_var := virtual_lin_s - (KERNEL_ORDER(conv_layer_index_s) - 1); -- carara
                                end if;
                                
                                k_ref_index_s <= new_k_ref_index_var;
                            else-- Feature map line generated
                                --  ENTRA AQUI QUANDO TERMINNA UMA LINHA DO FEATURE MAP
                                
                                fmap_lin_index_s <= fmap_lin_index_s + 1;
                                fmap_col_index_s <= 0;
                                new_k_ref_index_var := line_start_s + NEW_LINE(conv_layer_index_s); 
                                k_ref_index_s <= new_k_ref_index_var;
                                pool_start_s <= new_k_ref_index_var;
                                line_start_s <= new_k_ref_index_var;
                                new_virtual_col_var := -PAD(conv_layer_index_s);
                                new_virtual_lin_var := virtual_lin_s - 1;                                
                            
                                -- reset kernel line jump
                                currentState_s  <= FINISH_MAC;                                
                            end if;                                
                        end if;
                        
                        flat_rd_addr_s <= new_k_ref_index_var; 
                        virtual_col_s <= new_virtual_col_var; 
                        virtual_lin_s <= new_virtual_lin_var; 

                        k_line_jump_s <= CONV_INPUT_WIDTH(conv_layer_index_s);
                        k_col_index_s <= 0;
                        k_lin_index_s <= 0;
                         
                        -- reset kernel line jump
                        currentState_s  <= FINISH_MAC; 
                        
                        -- incrementa para endereçar o bias
                        filter_weight_line_addr_s <= filter_weight_line_addr_s + 1;

                    end if;

                -- Inference
                when FULL_CONNECTED =>
                
                    if fc_layer_index_s = 0 then -- Data read from feature map memories
                        if flat_rd_addr_s < NEURON_INPUTS_PER_CHANNEL - 1 then
                            flat_rd_addr_s <= flat_rd_addr_s + 1;
                        else 
                            flat_rd_addr_s <= 0; 
                        end if;
        
                        -- Go to the next input channel
                        if flat_rd_addr_s = 0 then 
                            if channel_index_s < CONV_LAYER_FILTERS(CONV_LAYERS - 1) - 1 then
                                channel_index_s <= channel_index_s + 1;
                                currentState_s <= FULL_CONNECTED;
                            else     
                                full_layer_weights_addr_s <= full_layer_weights_addr_s + 1;                                
                                flat_rd_addr_s <= 0; -- Restart the feature map memories read
                                currentState_s <= FC_ADD_BIAS;
                            end if;
                        end if;
                        
                    else -- Layers after neurons layer 0  
                         -- Data comes from NEURONS memory
                        if previous_layer_index_s < (NEURONS(fc_layer_index_s - 1) - 1) then
                            previous_layer_index_s <= previous_layer_index_s + 1;
							currentState_s <= FULL_CONNECTED;
                        else 
                            full_layer_weights_addr_s <= full_layer_weights_addr_s + 1;                            
                            currentState_s <= FC_ADD_BIAS;
                        end if;
						
						if previous_layer_index_s < (NEURONS(fc_layer_index_s - 1) - 1) then
							neurons_rd_addr_s <= neurons_rd_addr_s + 1;
						elsif neuron_index_s < (NEURONS(fc_layer_index_s) - 1) then
							neurons_rd_addr_s <= neurons_rd_addr_s - NEURONS(fc_layer_index_s - 1);
						end if;
                    end if;

                    -- Update weight memory address
                    if weight_sel_s = WEIGHTS_PER_LINE - 2 or
                        -- When the last weight memory word has less than WEIGHTS_PER_LINE weights and current full connected layer is the first reading the last input channel
                        (fc_layer_index_s = 0 and flat_rd_addr_s = NEURON_INPUTS_PER_CHANNEL - 1 and channel_index_s = CONV_LAYER_FILTERS(CONV_LAYERS - 1) - 1) or
                        -- When the last weight memory word has less than WEIGHTS_PER_LINE weights and current full connected layer is after the first
                        (fc_layer_index_s > 0 and previous_layer_index_s = NEURONS(fc_layer_index_s - 1) - 2) then     
                            full_layer_weights_addr_s <= full_layer_weights_addr_s + 1;
                    end if;

                    if weight_sel_s < WEIGHTS_PER_LINE - 1 then
                        weight_sel_s <= weight_sel_s + 1;
                    else
                        weight_sel_s <= 0;
                    end if;
                                      
                when FC_ADD_BIAS =>           
					currentState_s <= FC_MEM_WRITE;
					
				when FC_MEM_WRITE =>                    
                    if fc_layer_index_s = (FULL_LAYERS - 1) and neuron_index_s = NEURONS(fc_layer_index_s) - 1 then
                        currentState_s <= DONE;                    
                    else
                        
						
						if neuron_index_s < NEURONS(fc_layer_index_s) - 1 then
							neuron_index_s <= neuron_index_s + 1;                        
																
							if fc_layer_index_s = 0 then
								flat_rd_addr_s <= flat_rd_addr_s + 1;
							else
								-- neurons_rd_addr_s must point the second neuron 
								-- from previous layer when entering FULL_CONNECTED
								neurons_rd_addr_s <= neurons_rd_addr_s + 1;
							end if;
							
							currentState_s <= FULL_CONNECTED;
							
						elsif fc_layer_index_s < (FULL_LAYERS - 1) then
							fc_layer_index_s <= fc_layer_index_s + 1;
							neuron_index_s <= 0;
							previous_neurons <= NEURONS(fc_layer_index_s);
							currentState_s <= START_NEW_FC_LAYER;
						end if;	
					end if;                    
                    
                    weight_sel_s <= 0;
                    previous_layer_index_s <= 0;
                    channel_index_s <= 0;                   
                
                when START_NEW_FC_LAYER =>
					-- neurons_rd_addr_s must point the second neuron 
					-- from previous layer when entering FULL_CONNECTED
					neurons_rd_addr_s <= neurons_rd_addr_s + 1;
					currentState_s <= FULL_CONNECTED;
				
				when DONE =>
                    currentState_s <= INIT;

            end case;
        end if;

    end process;

    done_o <= '1' when currentState_delayed_s = DONE else '0';
    
    --##########################
    --#                        #
    --#       CONV LAYERS      #
    --#                        #
    --##########################

    -- Data to be multiplied (shift) by a weight
    -- Data from memory (data_i) signed extended
    SIGNED_INPUT: if SIGNED_DATA_I = true generate
        feature_s <= (others=>'0') when pad_data_s else
                     RESIZE(SIGNED(data_i), feature_s'length) when conv_layer_index_s = 0 else
                     RESIZE(SIGNED(fmap_o_s(channel_index_s)), feature_s'length);
    end generate;
    
    -- Data from memory (data_i) zero extended  
    UNSIGNED_INPUT: if SIGNED_DATA_I = false generate
        feature_s <= (others=>'0') when pad_data_s else
                     SIGNED(RESIZE(UNSIGNED(data_i), feature_s'length)) when conv_layer_index_s = 0 else
                     RESIZE(SIGNED(fmap_o_s(channel_index_s)), feature_s'length);
    end generate;
    
    fmap_wr_s   <= '1' when currentState_delayed_s = CONV_MEM_WRITE else '0';
    
    fmap_rd_addr_s <= TO_SIGNED(flat_rd_addr_s, FMAPS_ADDR_WIDTH); 

    -- 0 to 11 = 12 -> NUMERO DE KERNELS DA CAMADA QUE TEM MAIS KERNELS
    FILTERS_DATAPATH: for f in 0 to MAX_FILTERS - 1 generate      
        
        ----------------------------
        -- Weights/Signs memories --
        ----------------------------
        FILTER_WEIGHTS : entity work.Memory(BlockRAM)
        generic map (
            imageFileName   => FILTER_WEIGTHS_FILES(f),    --imageFileName     
            DATA_WIDTH      => WEIGHTS_MEM_DATA_WIDTH,
            ADDR_WIDTH      => FILTERS_ADDR_WIDTH
        )
        port map (
            clock           => clk,
            wr              => '0',
            write_address   => (others=>'0'),
            read_address    => STD_LOGIC_VECTOR(filter_weight_line_addr_s),
            data_i          => (others=>'0'),        
            data_o          => filter_weights_line_s(f)
        );

        FILTER_SIGNS : entity work.Memory(BlockRAM)
        generic map (
            imageFileName   => FILTER_SIGNS_FILES(f),      --imageFileName
            DATA_WIDTH      => WEIGHTS_PER_LINE,
            ADDR_WIDTH      => FILTERS_ADDR_WIDTH     
        )
        port map (
            clock           => clk,
            wr              => '0',
            write_address   => (others=>'0'),
            read_address    => STD_LOGIC_VECTOR(filter_weight_line_addr_s),
            data_i          => (others=>'0'),        
            data_o          => filter_signs_o_s(f)
        );

        -- Select one weight/sign from a weight/sign memory line        
        filter_weight_s(f) <= filter_weights_line_s(f)((WEIGHTS_PER_LINE - 1 - weight_sel_s) * WEIGHT_WIDTH + WEIGHT_WIDTH - 1 downto (WEIGHTS_PER_LINE - 1 - weight_sel_s) * WEIGHT_WIDTH);
        
        filter_sign_s(f) <= filter_signs_o_s(f)(WEIGHTS_PER_LINE - 1 - weight_sel_s);
        
        ----------------------------------------------
        -- Convolution multiplications (shift left) --
        ----------------------------------------------
		-- Computes the difference between the current weight and the scaling (cnn_weight_shift_s),
		-- in order to apply only one shift operation
		filter_weight_diff_s(f) <= STD_LOGIC_VECTOR(WEIGHT_SHIFT - UNSIGNED('0' & filter_weight_s(f)));        
       
	   
        conv_shift_s(f) <= SHIFT_RIGHT(feature_s, TO_INTEGER(UNSIGNED(filter_weight_diff_s(f)))) when filter_weight_diff_s(f)(filter_weight_diff_s(f)'left) = '0' else
                           SHIFT_LEFT(feature_s, TO_INTEGER(UNSIGNED(not filter_weight_diff_s(f)) + 1));
        
		-- Breaks the MAC in 2 stages
		--	1. Multiply (shift)
		--	2. Accumulate
		-- Delayed signals (1 cycle) are used as pipeline registers
		process(clk)
		begin
			if rising_edge(clk) then
				if SIGNED(filter_weight_s(f)) = 0 then	-- weight entre 0 e 1, arredondado para 0 aqui. log2(weight) negativo é marcado com -1.
					conv_shift_delayed_s(f) <= (others=>'0');
				else
					conv_shift_delayed_s(f) <= conv_shift_s(f);
				end if;
				filter_sign_delyed_s(f) <= filter_sign_s(f);
				filter_weights_line_delayed_s(f) <= filter_weights_line_s(f); -- Used when adding bias
			end if;
		end process;
        
		conv_mult_s(f) <= (not conv_shift_delayed_s(f)) + 1 when filter_sign_delyed_s(f) = '1' else -- Negative weight
                          conv_shift_delayed_s(f);
        
        -------------------------------
        -- Convolution accumulations --
        -------------------------------
        process(clk)
            variable  conv_acc_bias_var: conv_acc_array_t;
        begin
                        
            if rising_edge(clk) then  
                if currentState_delayed_s = INIT then
                    conv_acc_s(f) <= (others=>'0');
                    max_pool_s(f) <= (others=>'0');
                
                -- Accumulation
                elsif currentState_delayed_s = RECEPTIVE_FILED_CONV or currentState_delayed_s = FINISH_MAC then  
                    conv_acc_s(f) <= conv_acc_s(f) + conv_mult_s(f)(CONV_ACC_WIDTH - 1 downto 0);
                
				-- Add bias				
                elsif currentState_delayed_s = CONV_ADD_BIAS then
                    conv_acc_bias_var(f) := conv_acc_s(f) + SIGNED(filter_weights_line_delayed_s(f)(CONV_ACC_WIDTH - 1 downto 0));
                    
                    ----------
                    -- ReLU --       max_pool_s(f) initialized as 0
                    ---------- 
                    if conv_acc_bias_var(f) > max_pool_s(f) then
                        max_pool_s(f) <= conv_acc_bias_var(f);
                    end if;
                    
					-- Reset accumulator for the next MAC computation
                    conv_acc_s(f) <= (others=>'0');
                    
                elsif currentState_delayed_s = CONV_MEM_WRITE then
                    max_pool_s(f) <= (others=>'0');
                
                end if;           
            end if;
        end process;
        
        
        ---------------------------------------------------------
        -- Computed feature map to store at feature map memory --
        ---------------------------------------------------------
        fmap_i_s(f) <=  STD_LOGIC_VECTOR(max_pool_s(f));

        
        --------------------------
        -- Feature map memories --
        --------------------------
        FEATURE_MAP: entity work.Memory(BlockRAM)
        generic map (
            --imageFileName   => "mif/FeatureMap.mif",
            DATA_WIDTH      => CONV_ACC_WIDTH,
            ADDR_WIDTH      => FMAPS_ADDR_WIDTH             
        )
        port map (
            clock           => clk,
            wr              => fmap_wr_s,
            write_address   => STD_LOGIC_VECTOR(fmap_wr_addr_delayed_s),
            read_address    => STD_LOGIC_VECTOR(fmap_rd_addr_s),
            data_i          => fmap_i_s(f),        
            data_o          => fmap_o_s(f)
        );
        
    end generate FILTERS_DATAPATH;



    

    --###############################
    --#                             #
    --#    FULL CONNECTED LAYERS    #
    --#                             #
    --###############################

    neuron_input_s  <= RESIZE(SIGNED(fmap_o_s(channel_index_s)), neuron_input_s'length) when fc_layer_index_s = 0 else
                       RESIZE(SIGNED(neurons_o_s), neuron_input_s'length); -- Feature come from last full connected layer
    
	----------------------------
    -- Weights/Signs memories --
    ----------------------------
    FULL_LAYER_WEIGHTS : entity work.Memory(BlockRAM)
    generic map (
        imageFileName   => "raw/FullLayer_weights.txt",    --imageFileName 
        DATA_WIDTH      => WEIGHTS_MEM_DATA_WIDTH,
        ADDR_WIDTH      => FULL_LAYER_ADDR_WIDTH            
    )
    port map (
        clock           => clk,
        wr              => '0',
        write_address   => (others=>'0'),
        read_address    => STD_LOGIC_VECTOR(full_layer_weights_addr_s),
        data_i          => (others=>'0'),        
        data_o          => neuron_weights_line_s
    );

    FULL_LAYER_SIGNS : entity work.Memory(BlockRAM)
    generic map (
        imageFileName   => "raw/FullLayer_signs.txt",       --imageFileName
        DATA_WIDTH      => WEIGHTS_PER_LINE,
        ADDR_WIDTH      => FULL_LAYER_ADDR_WIDTH       
    )
    port map (
        clock           => clk,
        wr              => '0',
        write_address   => (others=>'0'),
        read_address    => STD_LOGIC_VECTOR(full_layer_weights_addr_s),
        data_i          => (others=>'0'),        
        data_o          => neuron_signs_line_s
    );
        
    -- Select one weight/sign from a weight/sign memory line        
    neuron_weight_s <= neuron_weights_line_s(WEIGHT_WIDTH - 1 + (WEIGHTS_PER_LINE - 1 - weight_sel_s) * WEIGHT_WIDTH downto (WEIGHTS_PER_LINE - 1 - weight_sel_s) * WEIGHT_WIDTH);
  
    neuron_sign_s <= neuron_signs_line_s(WEIGHTS_PER_LINE - 1 - weight_sel_s);
    
    -------------------------------------------------
    -- Full connected multiplications (shift left) --
    -------------------------------------------------
	
	-- Computes the difference between the current weight and the scaling (cnn_weight_shift_s),
	-- in order to apply only one shift operation
	neuron_weight_diff_s <= STD_LOGIC_VECTOR(WEIGHT_SHIFT - UNSIGNED('0' & neuron_weight_s));

    neuron_shift_s <= SHIFT_RIGHT(neuron_input_s, TO_INTEGER(UNSIGNED(neuron_weight_diff_s))) when neuron_weight_diff_s(neuron_weight_diff_s'left) = '0' else
                      SHIFT_LEFT(neuron_input_s, TO_INTEGER(UNSIGNED(not neuron_weight_diff_s) + 1));     

	-- Breaks the MAC in 2 stages
	--	1. Multiply (shift)
	--	2. Accumulate
	-- Delayed signals (1 cycle) are used as pipeline registers
	process(clk)
	begin
		if rising_edge(clk) then
			if SIGNED(neuron_weight_s) = 0 then  -- weight entre 0 e 1, arredondado para 0 aqui. log2(weight) negativo é marcado com -1.
				neuron_shift_delayed_s <= (others=>'0');
			else
				neuron_shift_delayed_s <= neuron_shift_s;
			end if;
	
			neuron_sign_delayed_s <= neuron_sign_s;
			neuron_weights_line_delayed_s <= neuron_weights_line_s; -- Used when adding bias
		end if;
	end process;

    -- Finalize the multiplication setting the sign (neuron_input_s * neuron_weight_s) 
	neuron_mult_s <= (not neuron_shift_delayed_s) + 1 when neuron_sign_delayed_s = '1' else
                     neuron_shift_delayed_s;
    
    ----------------------------------------
    -- Full connected layer accumulations --
    ----------------------------------------
    ACCUMULATOR: process(clk)
        variable temp : SIGNED(NEURON_ACC_WIDTH - 1 downto 0);
    begin 
        if rising_edge(clk) then
            -- Delayed currentState used to support pipelined MAC
			if currentState_delayed_s = INIT then
				neuron_acc_s <= (others=>'0');
				neurons_wr_addr_s <= (others=>'0');

            -- Accumulation
            elsif currentState_delayed_s = FULL_CONNECTED then                  
                neuron_acc_s <= neuron_acc_s + neuron_mult_s(NEURON_ACC_WIDTH - 1 downto 0);
			
			-- Add bias
            elsif currentState_delayed_s = FC_ADD_BIAS then
                temp := neuron_acc_s + RESIZE(SIGNED(neuron_weights_line_delayed_s), NEURON_ACC_WIDTH);
				
				-- ReLU applied only on hidden full connected layers
				if temp < 0 and fc_layer_index_s < (FULL_LAYERS - 1)  then
					neuron_acc_s <= (others=>'0');
				else
					neuron_acc_s <= temp;
				end if;			
			
			-- NEURONS memory store				
			elsif currentState_delayed_s = FC_MEM_WRITE then
				neurons_wr_addr_s <= neurons_wr_addr_s + 1;
				
				-- Searches for the higher output value
				if neuron_acc_index_delayed_s = 0 then
					neuron_max_s <= neuron_acc_s;
					class_s <= neuron_acc_index_delayed_s;
				
				elsif neuron_acc_s > neuron_max_s then
					neuron_max_s <= neuron_acc_s;
					class_s <= neuron_acc_index_delayed_s;
				end if;
				
				-- Reset accumulator for the next MAC computation
				neuron_acc_s <= (others=>'0');
            end if;
        end if;
    end process;	
	
	neurons_wr_s <= '1' when currentState_delayed_s = FC_MEM_WRITE else '0'; 
	
	-- Stores the neuron outputs from all full connected layers
	NEURONS: entity work.Memory(BlockRAM)
    generic map (
        DATA_WIDTH      => NEURON_ACC_WIDTH,
        ADDR_WIDTH      => NEURONS_ADDR_WIDTH             
    )
    port map (
        clock           => clk,
        wr              => neurons_wr_s,
        write_address   => STD_LOGIC_VECTOR(neurons_wr_addr_delayed_s),
        read_address    => STD_LOGIC_VECTOR(neurons_rd_addr_s),
        data_i          => STD_LOGIC_VECTOR(neuron_acc_s),        
        data_o          => neurons_o_s
    );


end Behavioral;