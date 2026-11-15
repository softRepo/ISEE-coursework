module mux_2to1 (out, in0, in1, addr);
    parameter n = 1;

    // Inputs
    input  [n-1:0] in0, in1;
    input addr;

    // Outputs
    output [n-1:0] out;

    // Assign value
    assign out = (addr) ? in1 : in0;
endmodule
