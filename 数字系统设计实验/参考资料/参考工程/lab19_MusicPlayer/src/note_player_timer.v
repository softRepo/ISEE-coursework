module note_player_timer(clk, beat, duration_to_load, timer_clear, timer_done);
    // Inputs and outputs
    input clk, beat, timer_clear;
    input [5:0] duration_to_load;
    output reg timer_done;

    reg [5:0] count;

    always @(posedge clk) begin
        if (timer_clear) // reset
            begin timer_done = 0; count = 0; end
        else if (beat)
        begin
            if (count == duration_to_load - 1)
                begin count = 0; timer_done = 1; end
            else
                count = count + 1;
        end
        else
            timer_done = 0;
    end
endmodule
