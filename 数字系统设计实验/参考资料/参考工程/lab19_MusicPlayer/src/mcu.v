module mcu(clk, reset, play_pause, next, play, reset_play, song_done, song);
    input clk, reset, play_pause, next, song_done;
    output play, reset_play;
    output[1:0] song;

    // Instantiate controller
    mcu_controller controller(.play_pause(play_pause), .reset(reset), .song_done(song_done), .next(next), .clk(clk), .play(play), .reset_play(reset_play), .next_song(next_song));

    // Instantiate counter
    mcu_counter counter(.en(next_song), .clk(clk), .clr(reset), .song(song));
endmodule
