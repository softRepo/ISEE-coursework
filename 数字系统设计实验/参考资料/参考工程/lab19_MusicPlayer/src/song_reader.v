module song_reader(clk, reset, play, song, song_done, note, duration, new_note, note_done);

    // Inputs and outputs
    input clk, reset, play, note_done;
    input [1:0] song;
    output song_done, new_note;
    output [5:0] note, duration;

    // Wires
    wire [6:0] addr;
    wire[11:0] dout;
    wire co;
    wire[4:0] q;
    assign note = dout[11:6];
    assign duration = dout[5:0];
    assign addr = {song, q}; // addr[6:5] = song[1:0], addr[4:0] = q[4:0]

    // Instantiate controller
    song_reader_controller controller(.clk(clk), .reset(reset), .note_done(note_done), .play(play), .new_note(new_note));

    // Instantiate address counter
    song_reader_addr_counter addr_counter(.clk(clk), .reset(reset), .note_done(note_done), .q(q), .co(co));

    // Instantiate song_rom
    song_rom rom(.clk(clk), .addr(addr), .dout(dout));

    // Instantiate finish module
    song_reader_done done(.clk(clk), .duration(duration), .co(co), .song_done(song_done));
endmodule
