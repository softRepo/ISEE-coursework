module dffre(d, en, r, clk, q);
    input d, en, r, clk;
    output reg q = 0;
    parameter n = 1;

    always @(posedge clk)
    begin
        if (r)
            q = 0;
        else if (en)
            q = d;
    end
endmodule
