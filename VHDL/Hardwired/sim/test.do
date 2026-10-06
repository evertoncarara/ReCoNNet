onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -radix hexadecimal /skynet_test_tb/test/CNN/done_o
add wave -noupdate -radix hexadecimal /skynet_test_tb/test/CNN/class_o
add wave -noupdate -radix hexadecimal -childformat {{/skynet_test_tb/test/CNN/NEURONS/memoryArray(0) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(1) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(2) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(3) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(4) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(5) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(6) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(7) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(8) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(9) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(10) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(11) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(12) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(13) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(14) -radix hexadecimal} {/skynet_test_tb/test/CNN/NEURONS/memoryArray(15) -radix hexadecimal}} -expand -subitemconfig {/skynet_test_tb/test/CNN/NEURONS/memoryArray(0) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(1) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(2) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(3) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(4) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(5) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(6) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(7) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(8) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(9) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(10) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(11) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(12) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(13) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(14) {-radix hexadecimal} /skynet_test_tb/test/CNN/NEURONS/memoryArray(15) {-radix hexadecimal}} /skynet_test_tb/test/CNN/NEURONS/memoryArray
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {135234375 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {0 ps} {157500 ns}
