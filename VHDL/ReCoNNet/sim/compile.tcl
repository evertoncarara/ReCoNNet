# TCL ModelSim compile script
# Pay atention on the compilation order!!!



# Sets the compiler
#set compiler vlog
set compiler vcom


# Creats the work library if it does not exist
if { ![file exist work] } {
    vlib work
}




#########################
### Source files list ###
#########################

# Source files listed in hierarchical order: botton -> top
set sourceFiles {   
    ../src/ReCoNNet_pkg.vhd
    ../src/Util_pkg.vhd    
    ../src/Memory.vhd
    ../src/ReCoNNet.vhd
    ResetSynchronizer.vhd
    UART_TX_v1.vhd
    UART_RX_v1.vhd
    AppMessage.vhd
    ReCoNNet_test.vhd
    ReCoNNet_test_tb.vhd
}




set top ReCoNNet_test_tb



###################
### Compilation ###
###################

if { [llength $sourceFiles] > 0 } {
    
    foreach file $sourceFiles {
        if [ catch {$compiler $file} ] {
            puts "\n*** ERROR compiling file $file :( ***" 
            return;
        }
    }
}




################################
### Lists the compiled files ###
################################

if { [llength $sourceFiles] > 0 } {
    
    puts "\n*** Compiled files:"  
    
    foreach file $sourceFiles {
        puts \t$file
    }
}


puts "\n*** Compilation OK ;) ***"

#vsim $top
set NumericStdNoWarnings 1

#do "set StdArithNoWarnings 1; set NumericStdNoWarnings 1"

