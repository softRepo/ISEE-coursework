module note_player(clk, reset, play_enable, note_to_load, duration_to_load, note_done, load_new_note, beat, sampling_pulse, sample, sample_ready);
    // Inputs and outputs
    input clk, reset, play_enable, load_new_note, sampling_pulse, beat;
    input [5:0] note_to_load, duration_to_load;
    output note_done, sample_ready;
    output [15:0] sample;

    // Wires
    wire [5:0] addr;
    wire [19:0] dout;
    wire timer_clear, timer_done, load;
    wire R = (~play_enable) || reset;

    // Instantiate d flip-flop
    dffre #(.n(6)) dffre1(.d(note_to_load), .en(load), .r(R), .clk(clk), .q(addr));

    // Instantiate frequency ROM
    frequency_rom freq_rom(.clk(clk), .dout(dout), .addr(addr));

    // Instantiate dds
    dds dds1(.clk(clk), .reset(R), .k({2'b00, dout}), .sampling_pulse(sampling_pulse), .new_sample_ready(sample_ready), .sample(sample));

    // Instantiate controller
    note_player_controller controller(.clk(clk), .reset(reset), .play_enable(play_enable), .load_new_note(load_new_note), .load(load), .timer_clear(timer_clear), .timer_done(timer_done), .note_done(note_done));

    // Instantiate note_player_timer
    note_player_timer timer(.clk(clk), .beat(beat), .duration_to_load(duration_to_load), .timer_clear(timer_clear), .timer_done(timer_done));
endmodule
