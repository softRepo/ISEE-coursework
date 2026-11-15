module song_reader_controller(clk, reset, note_done, play, new_note);
    input clk, reset, note_done, play;
    output reg new_note;

    // State encoding
    reg[1:0] state;
    parameter RESET = 0, NEW_NOTE = 1, WAIT = 2, NEXT_NOTE = 3;

    // State behaviour with clk
    always @(posedge clk) begin
        if (reset)
            begin state = RESET; new_note = 0; end
        else
            case (state)
                RESET: begin
                    if (play)
                        begin state = NEW_NOTE; new_note = 1; end
                end
                NEW_NOTE: begin state = WAIT; new_note = 0; end
                WAIT: begin
                    if (play)
                        begin
                            if (note_done)
                                begin state = NEXT_NOTE; new_note = 0; end
                        end
                    else
                        begin state = RESET; new_note = 0; end
                end
                NEXT_NOTE: begin state = NEW_NOTE; new_note = 1; end
            endcase
    end
endmodule
