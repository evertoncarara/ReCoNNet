
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;
use work.ReCoNNet_pkg.all;
 
 
ENTITY ReCoNNet_test IS
    generic (
        FILE_NAME       : string := "UNUSED"
    );
    port (
        rst_led     : out std_logic;
        done_led    : out std_logic; 
        load_input_led  : out std_logic;        
        clk         : in std_logic;   -- PIN_AJ16
        rst         : in std_logic;   -- PIN_AA26
        rx_i        : in std_logic;   -- PIN_B27
        tx_o        : out std_logic   -- PIN_H24
    );
END ReCoNNet_test;
 
ARCHITECTURE behavior OF ReCoNNet_test IS 
    signal done_s, start_tx_s, tx_active_s, rst_sync, rst_sync_n, data_av_s: std_logic;
    signal data_rx_s: std_logic_vector(7 downto 0);
    signal i: integer;
    signal max_s: SIGNED(NEURON_ACC_WIDTH - 1 downto 0);
    signal max_idx_s: std_logic_vector(7 downto 0);
    
    type State is (LOADING, START_CNN, WAIT_DONE, MAX_OUT, SEND_DATA, SENDING);
    signal currentState_s: State;
    
    --constant INPUT_SIZE : integer := 28 * 28;   -- MNIST/Fashion-MNINST
    constant INPUT_SIZE : integer := 32 * 32;   -- GTSRB/CIFAR10
    constant SIGNED_DATA_I : boolean := true;
    constant DATA_WIDTH : integer := 10;
    constant ADDR_WIDTH : integer := 11;
    
    signal read_address_s   : std_logic_vector(ADDR_WIDTH - 1 downto 0);
    signal write_address_s  : UNSIGNED(ADDR_WIDTH - 1 downto 0);
    signal feature_s        : std_logic_vector(11 downto 0); -- CIFAR10 shift = 7
    signal start_s, wr_s    : std_logic; 
    --signal linear_s, cnn_out_s: neuron_out_array_t;

    
 
    
BEGIN

    start_tx_s <= '1' when currentState_s = SEND_DATA else '0';
    
    RST_SYNCH: entity work.ResetSynchronizer 
    port map (
        clk         => clk,  
        rst_in      => rst,  
        rst_out     => rst_sync
    );
    
    rst_sync_n <= not rst_sync;
        
    rst_led <= rst_sync_n;
    
    --start_s <= '1' when currentState_s = START_CNN else '0';
    start_s <= '1'; -- DEBUG
    
    -- Instantiate the Unit Under Test (UUT)
    CNN: entity work.ReCoNNet    
    generic map ( -- Comment generic map parameters for Post Synthesis/Implementation simulation
        SIGNED_DATA_I   => SIGNED_DATA_I,
        DATA_WIDTH      => DATA_WIDTH,
        ADDR_WIDTH      => ADDR_WIDTH
    )
    port map (
        clk             => clk,
        rst             => rst_sync_n,
        start_i         => start_s,
        --linear_o        => linear_s,
        done_o          => done_s,
        load_input_led  => load_input_led,
        
        -- Memory interface
        data_i          => feature_s(9 downto 0),
        address_o       => read_address_s,
        
        wr_params_i     => '0',
        wr_filter_weights_i => '0',
        wr_filter_signs_i => '0',
        filter_weights_i => (others=>'0'),
        filter_signs_i => (others=>'0'),
        wr_fc_weights_i => '0', 
        wr_fc_signs_i => '0',
        data_conf_i => (others=>'0'),
        address_i => (others=>'0')
    );


    wr_s <= data_av_s when currentState_s = LOADING else '0';
    
    FEATURE_RAM: entity work.Memory(BlockRAM)
    generic map (  
        imageFileName   => FILE_NAME,
        --DATA_WIDTH      => 8,
        DATA_WIDTH      => 12, -- CIFAR10 shift = 7. 12 para fechar 3 chars hexa na imagem
        ADDR_WIDTH      => ADDR_WIDTH
    )
    port map (
        clock           => clk,
        wr              => wr_s,
        write_address   => STD_LOGIC_VECTOR(write_address_s),
        read_address    => read_address_s,
        --data_i          => data_rx_s,         
        data_i          => (others=>'0'),
        data_o          => feature_s
    );

    TX_TO_HOST: entity work.UART_TX_v1
    generic map(
        g_CLKS_PER_BIT    => FREQ_BAUD_RATE
    )
    port map (
        i_Clk       => clk,
        i_TX_DV     => start_tx_s,
        i_TX_Byte   => max_idx_s,
        o_TX_Serial => tx_o,
        o_TX_Active => tx_active_s
    );

    

    RX_FROM_HOST: entity work.UART_RX_v1
    generic map(
        g_CLKS_PER_BIT    => FREQ_BAUD_RATE
    )
    port map (
        i_Clk       => clk,
        i_RX_Serial => rx_i,
        o_RX_DV     => data_av_s,
        o_RX_Byte   => data_rx_s  
    );

   
    
    process (clk, rst_sync_n)
    begin
        if rst_sync_n = '1' then
            done_led <= '0';
            currentState_s <= WAIT_DONE; -- DEBUG
            --currentState_s <= LOADING;
            write_address_s <= (others=>'0');
        
        elsif rising_edge(clk) then
            case currentState_s is
                when LOADING =>
                    if data_av_s = '1' then
                        write_address_s <= write_address_s + 1;
                    end if;
                    
                    if write_address_s = INPUT_SIZE then
                        currentState_s <= START_CNN;
                        write_address_s <= (others=>'0');
                    end if;
                    
                    i <= 0;
                    done_led <= '0';
                
                when START_CNN =>
                    currentState_s <= WAIT_DONE;
                
                when WAIT_DONE =>
                    if done_s = '1' then  
                        done_led <= '1';
                        --max_s <= SIGNED(linear_s(i));
                        max_idx_s <= x"00";
                        --cnn_out_s <= linear_s;
                        i <= i + 1;
                        currentState_s <= MAX_OUT;
                    end if;
                    
                when MAX_OUT =>                                            
                    --if SIGNED(cnn_out_s(i)) > max_s then
                    --    max_s <= SIGNED(cnn_out_s(i));
                    --    max_idx_s <= STD_LOGIC_VECTOR(TO_UNSIGNED(i,8));
                    --end if;
                    --
                    --if i = cnn_out_s'length - 1 then
                    --    currentState_s <= SEND_DATA;
                    --else
                    --    i <= i + 1;
                    --end if;                   
                    
                when SEND_DATA =>
                    currentState_s <= SENDING;                
                    
                when SENDING =>
                    if tx_active_s = '0' then
                        currentState_s <= LOADING;
                    end if;               
                    
            end case;
        end if;
    end process;
END;
