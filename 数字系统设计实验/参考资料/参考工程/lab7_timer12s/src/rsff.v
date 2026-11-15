module rsff(clk, r, s, q);
    input clk, r, s;
    output reg q = 0;

    always @(posedge clk)
    begin
        if (r)
          q = 0;
        else if (s)
          q = 1;
    end
endmodule
