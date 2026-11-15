module counter_n(clk, r, en, co, q);
    parameter counter_bits = 1, n = 1;
    input clk, r, en;
    output co;
    output reg[counter_bits-1: 0] q = 0;

    always @(posedge clk)
    begin
        if (r)
            q = 0;
        else if (en)
            q = (q + 1) % n;
    end

    assign co = en && (q == n - 1);
endmodule
