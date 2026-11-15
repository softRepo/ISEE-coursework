module note_player_controller(clk, reset, play_enable, load_new_note, load, timer_clear, timer_done, note_done);
    // Inputs and outputs
    input clk, reset, play_enable, load_new_note, timer_done;
    output reg load, timer_clear, note_done;

    // State encoding
    reg[1:0] state;
    parameter RESET = 0, WAIT = 1, DONE = 2, LOAD = 3;

    // State behaviour with clk
    always @(posedge clk) begin
        if (reset) begin state = RESET; timer_clear = 1; load = 0; note_done = 0; end
        else
            case(state)
                RESET: begin state = WAIT; timer_clear = 0; load = 0; note_done = 0; end
                WAIT: begin
                    if (play_enable)
                        begin
                            if (timer_done) begin
                                state = DONE; timer_clear = 1; load = 0; note_done = 1;
                            end
                            else if (load_new_note) begin
                                state = LOAD; timer_clear = 1; load = 1; note_done = 0;
                            end
                            else begin
                                state = WAIT; timer_clear = 0; load = 0; note_done = 0;
                            end
                        end
                    else begin state = RESET; timer_clear = 1; load = 0; note_done = 0; end
                end
                DONE: begin state = WAIT; timer_clear = 0; load = 0; note_done = 0; end
                LOAD: begin state = WAIT; timer_clear = 0; load = 0; note_done = 0; end
            endcase
    end
endmodule
