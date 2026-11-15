module ModeComparator(a, b, m, y);
    // Inputs
    input [7:0] a, b;
    input m; // Notice that m is 1 bit

    // Outputs
    output [7:0] y;

    // Instantiate comparator
    wire agb;
    comp #(.n(8)) comp_inst(.a(a), .b(b), .agb(agb), .aeb(), .alb());

    // Instantiate mux_2to1
    wire[7:0] max, min;
    mux_2to1 #(.n(8)) mux1(.out(max), .in0(a), .in1(b), .addr(~agb));
    mux_2to1 #(.n(8)) mux2(.out(min), .in0(a), .in1(b), .addr(agb));
    mux_2to1 #(.n(8)) mux3(.out(y), .in0(max), .in1(min), .addr(m));
endmodule
