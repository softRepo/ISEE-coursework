module synchronizer(in, clk, out);
    input in, clk;
    output out;

    wire q1, q2;
    assign out = q1 && ~q2;

    // Instantiate d flip-flops
    dffre dffre1(.d(in), .en(1'b1), .r(1'b0), .clk(clk), .q(q1));
    dffre dffre2(.d(q1), .en(1'b1), .r(1'b0), .clk(clk), .q(q2));
endmodule
