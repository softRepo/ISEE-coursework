module frequency_divider(clk, ci, co);
    parameter n = 1, counter_bits = 1;
    input clk, ci;
    output reg co;

    reg [counter_bits - 1:0] count = 0;

    always @(posedge clk) begin
        if (ci) begin
            if (count == n - 1) begin
                count = 0;
                co = 1;
            end
            else
                count = count + 1;
        end
        else
            co = 0;
    end
endmodule
