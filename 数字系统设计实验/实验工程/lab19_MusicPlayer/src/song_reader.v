`timescale 1ns / 1ps

// 乐曲读取模块：从 song_rom 逐个读取 {note, duration}，并向 note_player 请求播放。
module song_reader (
    input  wire       clk,
    input  wire       reset,
    input  wire       play,
    input  wire [1:0] song,
    input  wire       note_done,
    output reg        song_done,
    output reg [5:0]  note,
    output reg [5:0]  duration,
    output reg        new_note
);
    localparam ST_IDLE     = 3'd0;
    localparam ST_FETCH    = 3'd1;
    localparam ST_WAIT_ROM = 3'd2;
    localparam ST_CHECK    = 3'd3;
    localparam ST_NEW_NOTE = 3'd4;
    localparam ST_WAIT     = 3'd5;
    localparam ST_DONE     = 3'd6;

    reg [2:0] state;
    reg [4:0] note_index;
    reg [6:0] rom_addr;
    wire [11:0] rom_data;

    song_rom song_rom_inst (
        .clk(clk),
        .addr(rom_addr),
        .dout(rom_data)
    );

    // song 选择高 2 位地址，note_index 选择一首歌中的 32 个音符位置。
    always @(posedge clk) begin
        if (reset) begin
            state <= ST_IDLE;
            note_index <= 5'd0;
            rom_addr <= 7'd0;
            note <= 6'd0;
            duration <= 6'd0;
            new_note <= 1'b0;
            song_done <= 1'b0;
        end else begin
            new_note <= 1'b0;
            song_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    note_index <= 5'd0;
                    if (play) begin
                        rom_addr <= {song, 5'd0};
                        state <= ST_WAIT_ROM;
                    end
                end

                ST_FETCH: begin
                    rom_addr <= {song, note_index};
                    state <= ST_WAIT_ROM;
                end

                ST_WAIT_ROM: begin
                    state <= ST_CHECK;
                end

                ST_CHECK: begin
                    // duration 为 0 表示当前曲目结束，输出一个周期 song_done。
                    state <= (rom_data[5:0] == 6'd0) ? ST_DONE : ST_NEW_NOTE;
                end

                ST_NEW_NOTE: begin
                    note <= rom_data[11:6];
                    duration <= rom_data[5:0];
                    new_note <= 1'b1;
                    state <= ST_WAIT;
                end

                ST_WAIT: begin
                    if (!play) begin
                        state <= ST_IDLE;
                    end else if (note_done) begin
                        note_index <= note_index + 1'b1;
                        state <= (&note_index) ? ST_DONE : ST_FETCH;
                    end
                end

                ST_DONE: begin
                    song_done <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule

