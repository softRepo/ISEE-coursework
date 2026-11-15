module song_reader_done(duration, co, clk, song_done);
    input [5:0] duration;
    input co;
    input clk;
    output reg song_done;

    always @(posedge clk) begin
        if (co || duration == 0)
            song_done = 1;
        else
            song_done = 0;
    end
endmodule
