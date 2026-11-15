module music_player(clk, reset, play_pause, next, NewFrame, sample, play, song);
    parameter sim = 0; // controls the frequency divider

    // Inputs and outputs
    input clk, reset, play_pause, next, NewFrame;
    output [15:0] sample;
    output play;
    output [1:0] song;

    // Wires
    wire reset_play, song_done, new_note, beat, note_done, ready;
    wire [5:0] duration_for_note, note_to_play;

    // Instantiate mcu
    mcu mcu_inst(.clk(clk),
		.reset(reset),
		.play_pause(play_pause),
		.next(next),
		.play(play),
		.song(song),
		.reset_play(reset_play),
		.song_done(song_done));

    // Instantiate song reader
    song_reader song_reader_inst (
		.clk(clk),
		.reset(reset_play),
		.play(play),
		.song(song),
		.song_done(song_done),
		.note(note_to_play),
		.duration(duration_for_note),
		.new_note(new_note),
		.note_done(note_done)
	);

    // Instantiate frequency divider
    frequency_divider #(.n(sim?64:1000), .counter_bits(sim?6:10)) freq_divider(.clk(clk), .ci(ready), .co(beat));

    // Instantiate note player
    note_player note_player_inst(.clk(clk),
		.reset(reset_play),
		.play_enable(play),
		.note_to_load(note_to_play),
		.duration_to_load(duration_for_note),
		.note_done(note_done),
		.load_new_note(new_note),
		.beat(beat),
		.sampling_pulse(ready),
		.sample(sample),
		.sample_ready());

    // Instantiate synchronizer
    synchronizer sync(.in(NewFrame), .clk(clk), .out(ready));
endmodule
