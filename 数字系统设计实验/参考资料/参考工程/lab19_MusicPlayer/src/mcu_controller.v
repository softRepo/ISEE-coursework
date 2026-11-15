module mcu_controller(play_pause, reset, song_done, next, clk, play, reset_play, next_song);
    input play_pause, reset, song_done, next, clk;
    output reg play, reset_play, next_song;

    // State encoding
    reg[1:0] state;
    parameter RESET = 0, PAUSE = 1, PLAY = 2, NEXT = 3;

    // State behaviour with clk
    always @(posedge clk) begin
        if (reset) begin state = RESET; play = 0; next_song = 0; reset_play = 1; end
        else
            case(state)
                RESET: begin state = PAUSE; play = 0; next_song = 0; reset_play = 0; end
                PAUSE: begin
                    if (play_pause)
                        begin state = PLAY; play = 1; next_song = 0; reset_play = 0; end
                    else if (next)
                        begin state = NEXT; play = 0; next_song = 1; reset_play = 1; end
                end
                PLAY: begin
                    if (play_pause)
                        begin state = PAUSE; play = 0; next_song = 0; reset_play = 0; end
                    else if (next)
                        begin state = NEXT; play = 0; next_song = 1; reset_play = 1; end
                    else if (song_done)
                        begin state = RESET; play = 0; next_song = 0; reset_play = 1; end
                end
                NEXT: begin state = PLAY; play = 1; next_song = 0; reset_play = 0; end
            endcase
    end
endmodule
