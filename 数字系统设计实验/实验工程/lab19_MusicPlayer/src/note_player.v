`timescale 1ns / 1ps

// 音符播放模块：按 duration_to_load 个 beat 播放一个音符，并输出 DDS 正弦采样。
module note_player (
    input  wire       clk,
    input  wire       reset,
    input  wire       play_enable,
    input  wire [5:0] note_to_load,
    input  wire [5:0] duration_to_load,
    input  wire       load_new_note,
    input  wire       beat,
    input  wire       sampling_pulse,
    output reg        note_done,
    output wire [15:0] sample,
    output wire       sample_ready
);
    reg  [5:0] note_reg;
    reg  [5:0] duration_reg;
    reg  [5:0] beat_count;
    wire [19:0] freq_word;
    wire [21:0] dds_k;

    frequency_rom frequency_rom_inst (
        .clk(clk),
        .addr(note_reg),
        .dout(freq_word)
    );

    assign dds_k = {2'b00, freq_word};

    dds dds_inst (
        .clk(clk),
        .reset(reset || !play_enable),
        .k(dds_k),
        .sampling_pulse(sampling_pulse),
        .new_sample_ready(sample_ready),
        .sample(sample)
    );

    // 新音符到来时装载 note/duration；每个 beat 计数一次，计满后产生 note_done。
    always @(posedge clk) begin
        if (reset || !play_enable) begin
            note_reg <= 6'd0;
            duration_reg <= 6'd0;
            beat_count <= 6'd0;
            note_done <= 1'b0;
        end else begin
            note_done <= 1'b0;

            if (load_new_note) begin
                note_reg <= note_to_load;
                duration_reg <= duration_to_load;
                beat_count <= 6'd0;
            end else if (beat && (duration_reg != 6'd0)) begin
                if (beat_count + 1'b1 >= duration_reg) begin
                    duration_reg <= 6'd0;
                    beat_count <= 6'd0;
                    note_done <= 1'b1;
                end else begin
                    beat_count <= beat_count + 1'b1;
                end
            end
        end
    end
endmodule

