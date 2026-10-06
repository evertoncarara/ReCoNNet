
-------------------------------------------------------------------------
-- Design unit: Memory
-- Description: Parameterizable data and address bus
-------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all; 
use std.textio.all;
use work.Util_pkg.all;


entity Memory is
    generic (
        SIZE            : integer   := 100;    -- Memory depth
        imageFileName   : string    := "UNUSED"; -- Memory content to be loaded
        DATA_WIDTH      : integer   := 8;
        ADDR_WIDTH      : integer   := 8
    );
    port (  
        clock           : in std_logic;
        wr              : in std_logic;      -- Write enable
        write_address   : in std_logic_vector(ADDR_WIDTH - 1 downto 0);
        read_address    : in std_logic_vector(ADDR_WIDTH - 1 downto 0); 
        data_i          : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        data_o          : out std_logic_vector(DATA_WIDTH - 1 downto 0)
    );
end Memory;

architecture BlockRAM of Memory is
    
    type Memory is array (0 to 2**ADDR_WIDTH - 1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    --type Memory is array (0 to SIZE-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    
    impure function MemoryLoad (imageFileName : in string) return Memory is
        FILE imageFile : text;
        variable fileLine : line;
        variable memoryArray : Memory;
        variable data_str : string(1 to DATA_WIDTH/4);
        
        variable i : natural := 0;
    begin 
        --if imageFileName /= "UNUSED" then
        --    file_open(imageFile, imageFileName, read_mode);
        --    for i in Memory'range loop
        --        readline (imageFile, fileLine);
        --        read (fileLine, data_str);
        --        memoryArray(i) := HexStringToStdLogicVector(data_str, DATA_WIDTH/4);
        --    end loop;
        --    file_close(imageFile);
        --end if;
        
        if imageFileName /= "UNUSED" then
            file_open(imageFile, imageFileName, read_mode);
            while NOT (endfile(imageFile)) loop
                readline (imageFile, fileLine);
                read (fileLine, data_str);
                memoryArray(i) := HexStringToStdLogicVector(data_str, DATA_WIDTH/4);
                i := i + 1;
            end loop;
            file_close(imageFile);
        end if;
        
        return memoryArray;
    end function;
    
    impure function MemoryLoadMIF(imageFileName: in string) return Memory is
        
        file imageFile : TEXT open READ_MODE is imageFileName;
        variable memoryArray: Memory;
        variable fileLine    : line;            -- Stores a read line from a text file
        variable data, addr  : string(1 to 32);  
        variable line_str    : string(1 to 100);
        variable char        : character;       -- Stores a single character
        variable i, j: integer;
        variable address: integer; 
        
        begin
            if imageFileName /= "UNUSED" then
            
                -- Searches for the "BEGIN" keyword
                while NOT (endfile(imageFile)) loop
                    
                    readline(imageFile, fileLine);
                    
                    assert fileLine'length < line_str'length;  -- Make sure line is big enough
                    
                    line_str := (others => ' '); -- Make sure that the previous line is overwritten
                    
                    if fileLine'length > 0 then -- Skip blank lines
                        read(fileLine, line_str(1 to fileLine'length));
                        
                        if line_str(1 to 5) = "BEGIN" then
                            report "Achou o BEGIN!!!";
                            exit;   -- BEGIN found
                        end if;
                    end if;
    
                end loop;
                
                -- Main loop to read the file
                -- Read addresses and data
                while NOT (endfile(imageFile)) loop    
                    
                    readline(imageFile, fileLine);
                    
                    -- Skip blank lines
                    if fileLine'length < 1 then
                        next;
    
                    elsif fileline(1 to 3) = "END" then
                        -- End of data
                        return memoryArray;
                    end if;
                
                
                    i := 1;
                    -- Read address character by character
                    loop
                        read(fileLine, char);
                        if char = ':' then  -- Separator (address:data)
                            address := integer'value(addr(1 to i - 1));
                            --report "address: " & integer'image(address);
                            --report "addr to int: " & integer'image(integer'value(addr(1 to i - 1)));
                            exit;
                        else
                            addr(i) := char;
                            i := i + 1;
                        end if;
                    end loop;
                    
                    
                    j := 1;                
                    -- Read data character by character
                    loop
                        read(fileLine, char);
                        if char = ';' then -- End of line
                            --report "addr: " & addr;
                            --report "data: " & data;
                            memoryArray(address) := HexStringToStdLogicVector(data, j - 1);                    
                            exit;
                        else
                            data(j) := char;
                            j := j + 1;
                        end if;
                    end loop;
                    
                end loop;
            end if;
            
            return memoryArray;
            
    end function;
    
    signal memoryArray : Memory := MemoryLoad(imageFileName);
    --signal memoryArray : Memory := MemoryLoadMIF(imageFileName);
    
    --signal memoryArray: memory;
    --attribute ram_init_file : string;
    --attribute ram_init_file of memoryArray: signal is imageFileName;
            
    signal wAddress : integer;
    signal rAddress : integer;
    
begin
       
    wAddress <= TO_INTEGER(UNSIGNED(write_address));
    rAddress <= TO_INTEGER(UNSIGNED(read_address));
       
    -- Process to control the memory access
    process(clock)
    begin
        if rising_edge(clock) then    -- Memory writing 
            if wr = '1' then
                memoryArray(wAddress) <= data_i; 
            end if;
        data_o <= memoryArray(rAddress);
        end if;   
    end process;
    
end BlockRAM;
