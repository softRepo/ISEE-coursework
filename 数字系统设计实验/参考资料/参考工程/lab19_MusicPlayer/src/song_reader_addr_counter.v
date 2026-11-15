module song_reader_addr_counter(clk, reset, note_done, q, co);
    input clk, reset, note_done;
    output reg[4:0] q;
    output reg co;

    always @(posedge clk) begin
        if (reset)
            begin co = 0; q = 0; end
        else if (note_done) begin
            if (q == 31)
                begin co = 1; q = 0; end
            else
                begin co = 0; q = q + 1; end
        end
    end
endmodule
