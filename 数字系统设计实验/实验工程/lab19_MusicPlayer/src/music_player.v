`timescale 1ns / 1ps

// 音乐播放器次顶层：连接主控制器、乐曲读取、音符播放、ready 同步和 beat 分频。
module music_player #(
    parameter sim = 0
)(
    input  wire       clk,
    input  wire       reset,
    input  wire       play_pause,
    input  wire       next,
    input  wire       NewFrame,
    output wire [15:0] sample,
    output wire       play,
    output wire [1:0] song
);
    localparam BEAT_DIVIDER = sim ? 64 : 1000;
    localparam BEAT_BITS    = sim ? 6  : 10;

    wire reset_play;
    wire song_done;
    wire note_done;
    wire new_note;
    wire [5:0] note;
    wire [5:0] duration;
    wire ready;
    wire beat;
    wire sample_ready;

    // NewFrame 来自音频接口时钟域，先同步到系统时钟域生成 48 kHz ready。
    pulse_sync ready_sync_inst (
        .clk(clk),
        .reset(reset),
        .in(NewFrame),
        .out(ready)
    );

    // ready 每 1000 次产生一次 beat，即 48 kHz / 1000 = 48 Hz。
    pulse_divider #(
        .n(BEAT_DIVIDER),
        .counter_bits(BEAT_BITS)
    ) beat_divider_inst (
        .clk(clk),
        .reset(reset),
        .enable(ready),
        .pulse(beat)
    );

    mcu mcu_inst (
        .clk(clk),
        .reset(reset),
        .play_pause(play_pause),
        .next(next),
        .song_done(song_done),
        .play(play),
        .reset_play(reset_play),
        .song(song)
    );

    song_reader song_reader_inst (
        .clk(clk),
        .reset(reset || reset_play),
        .play(play),
        .song(song),
        .note_done(note_done),
        .song_done(song_done),
        .note(note),
        .duration(duration),
        .new_note(new_note)
    );

    note_player note_player_inst (
        .clk(clk),
        .reset(reset || reset_play),
        .play_enable(play),
        .note_to_load(note),
        .duration_to_load(duration),
        .load_new_note(new_note),
        .beat(beat),
        .sampling_pulse(ready),
        .note_done(note_done),
        .sample(sample),
        .sample_ready(sample_ready)
    );
endmodule

