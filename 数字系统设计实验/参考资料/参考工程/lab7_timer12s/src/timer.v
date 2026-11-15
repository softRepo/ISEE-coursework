module timer(clk, r, en, done);
    parameter counter_bits = 1, n = 1;
    input clk, r, en;
    output done;
    reg [counter_bits-1:0] q = 0;

    always @(posedge clk)
    begin
        if (r)
            q = 0;
        else if (en)
            q = q + 1;
    end
    assign done = en && (q == n-1);
endmodule
