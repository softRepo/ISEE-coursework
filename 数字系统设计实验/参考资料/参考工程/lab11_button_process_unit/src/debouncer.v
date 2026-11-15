module debouncer(reset, in, clk, out);
    input in, clk, reset;
    output out;
    parameter sim = 0;

    wire pulse;

    frequency_divider #(.sim(sim)) freq_inst(#(.clk(clk), .out(pulse)));

endmodule
