module dffre(d, en, r, clk, q);
    parameter n = 1;
    input en, r, clk;
    input [n-1:0] d;
    output reg[n-1:0] q = 0;

    always @(posedge clk)
    begin
        if (r)
            q = 0;
        else if (en)
            q = d;
    end
endmodule
