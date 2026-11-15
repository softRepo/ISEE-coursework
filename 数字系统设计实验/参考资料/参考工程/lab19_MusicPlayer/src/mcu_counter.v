module mcu_counter(en, clk, clr, song);
    input en, clk, clr;
    output reg[1:0] song;

    always @(posedge clk) begin
        if (clr)
            song = 0;
        else if (en)
            song = song + 1;
    end
endmodule
