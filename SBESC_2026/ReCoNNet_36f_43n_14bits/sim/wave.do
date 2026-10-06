onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -format Logic -radix decimal /reconnet_test_tb/test/cnn/done_o
add wave -noupdate -format Literal -radix decimal /reconnet_test_tb/test/cnn/class_o
add wave -noupdate -format Literal -radix hexadecimal /reconnet_test_tb/test/cnn/neurons/memoryarray
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {1417549 ns} 0}
configure wave -namecolwidth 178
configure wave -valuecolwidth 148
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
configure wave -timelineunits ns
update
WaveRestoreZoom {0 ns} {4305 us}
