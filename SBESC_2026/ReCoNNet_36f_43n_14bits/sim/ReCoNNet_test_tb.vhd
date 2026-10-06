
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all; 
use work.ReCoNNet_pkg.all;

 
ENTITY ReCoNNet_test_tb IS
END ReCoNNet_test_tb;
 
ARCHITECTURE behavior OF ReCoNNet_test_tb IS 

    constant FREQ_BAUD_RATE     : integer := 868; -- 115200 baud rate at 100MHz
    
    -- Input memory file image
    constant FILE_NAME          : string := "input_int.txt";
  
    -- Inputs
    signal clk : std_logic := '0';
    signal tx, rx, push_button_rst, msg_start, msg_rst : std_logic;
    
    -- Clock period definition
    constant clk_period : time := 5 ns;  -- 200MHz
    
    signal ready              : std_logic;
    signal data_av            : std_logic;
    signal app_data_out       : std_logic_vector(7 downto 0);
 
BEGIN
    -- Terasic: push = 0; release = 1
    -- Atlys
    --      Red push-button (reset): push = 0; release = 1
    --      Other push-buttons: push = 1; release = 0

    clk <= not clk after clk_period / 2;
    
    push_button_rst <= '1', '0' after 4 * clk_period, '1' after 6 * clk_period;
    
    msg_start <= '0', '1' after 10 * clk_period, '0' after 12 * clk_period,
                 '1' after 89804070 ns, '0' after 89804080 ns;
    
    msg_rst <= '0', '1' after 4 * clk_period, '0' after 6 * clk_period,
               '1' after 89804050 ns, '0' after 89804060 ns;
 
    -- Instantiate the Unit Under Test (UUT)
    test: entity work.ReCoNNet_test
    generic map(
        FILE_NAME    => FILE_NAME
    )    
    port map (
        clk     => clk,
        rst     => push_button_rst,
        rx_i    => tx,
        tx_o    => rx
    );
   

    RX_FROM_NET: entity work.UART_RX_v1
    generic map(
        g_CLKS_PER_BIT    => FREQ_BAUD_RATE
    )
    port map (
        i_Clk         => clk,
        i_RX_Serial   => rx
    );


    MESSAGER0: entity work.AppMessage(behavioral)
    generic map(
        FILE_NAME    => FILE_NAME
    )
    port map(
        clk          => clk,
        rst          => msg_rst,
        start_signal => msg_start,
        data_out     => app_data_out,
        data_av      => data_av,
        ack          => ready
    );


    TX_TO_NET: entity work.UART_TX_v1
    generic map(
        g_CLKS_PER_BIT    => FREQ_BAUD_RATE
    )
    port map (
        i_Clk       => clk,
        i_TX_DV     => data_av,
        i_TX_Byte   => app_data_out,
        o_TX_Serial => tx,
        o_TX_Done   => ready
    );


END;
