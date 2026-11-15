`timescale 1ns / 1ps

// 主控制器：play_pause 在播放和暂停之间切换，next 切换到下一首曲目。
module mcu (
    input  wire       clk,
    input  wire       reset,
    input  wire       play_pause,
    input  wire       next,
    input  wire       song_done,
    output reg        play,
    output reg        reset_play,
    output reg [1:0]  song
);
    localparam ST_RESET = 2'd0;
    localparam ST_PAUSE = 2'd1;
    localparam ST_PLAY  = 2'd2;
    localparam ST_NEXT  = 2'd3;

    reg [1:0] state;

    // RESET 后进入暂停；NEXT 状态只保持一个周期，用于产生复位和切歌脉冲。
    always @(posedge clk) begin
        if (reset) begin
            state <= ST_RESET;
            song <= 2'd0;
        end else begin
            case (state)
                ST_RESET: state <= ST_PAUSE;

                ST_PAUSE: begin
                    if (next) begin
                        state <= ST_NEXT;
                        song <= song + 1'b1;
                    end else if (play_pause) begin
                        state <= ST_PLAY;
                    end
                end

                ST_PLAY: begin
                    if (next) begin
                        state <= ST_NEXT;
                        song <= song + 1'b1;
                    end else if (play_pause || song_done) begin
                        state <= song_done ? ST_RESET : ST_PAUSE;
                    end
                end

                ST_NEXT: state <= ST_PLAY;

                default: state <= ST_RESET;
            endcase
        end
    end

    // 组合输出：play 为播放使能，reset_play 用于同步复位 song_reader 和 note_player。
    always @(*) begin
        play = 1'b0;
        reset_play = 1'b0;
        case (state)
            ST_RESET: reset_play = 1'b1;
            ST_PLAY:  play = 1'b1;
            ST_NEXT:  reset_play = 1'b1;
        endcase
    end
endmodule

