# Run with Vivado: vivado -mode batch -source create_project.tcl
set root [file dirname [file normalize [info script]]]
create_project lab15_dds [file join $root build] -part xc7a200tfbg484-1
set_property target_language Verilog [current_project]
set design_files {}
set test_files {}
foreach f [glob -nocomplain [file join $root src *.v]] {
    if {[string match *_tb* [file tail $f]]} {
        lappend test_files $f
    } elseif {![string match *_func_synth.v [file tail $f]]} {
        lappend design_files $f
    }
}
if {[llength $design_files]} {add_files -norecurse $design_files}
if {[llength $test_files]} {add_files -fileset sim_1 -norecurse $test_files}
foreach f [glob -nocomplain [file join $root src *.edf]] {add_files -norecurse $f}
foreach f [glob -nocomplain [file join $root src *.xdc]] {add_files -fileset constrs_1 -norecurse $f}
set_property top dds [get_filesets sources_1]
update_compile_order -fileset sources_1
