`timescale 1ns / 1ps

// 直接数字频率合成器 DDS。
// k 为 22 位相位增量；sampling_pulse 每来一个周期，请求产生一个新的正弦采样。
module dds (
    input  wire        clk,
    input  wire        reset,
    input  wire [21:0] k,
    input  wire        sampling_pulse,
    output reg         new_sample_ready,
    output reg  [15:0] sample
);
    // 相位累加器的高 12 位表示一个完整正弦周期的位置，高 2 位区分四个象限。
    reg  [21:0] phase_acc;
    reg  [9:0]  rom_addr;
    reg  [1:0]  area_stage1;
    reg  [1:0]  area_stage2;
    reg         valid_stage1;
    reg         valid_stage2;
    wire [15:0] raw_data;

    wire [21:0] next_phase = phase_acc + k;
    wire [11:0] sine_index = next_phase[21:10];
    wire [9:0]  index_low  = sine_index[9:0];

    sine_rom sine_rom_inst (
        .clk(clk),
        .addr(rom_addr),
        .dout(raw_data)
    );

    // ROM 为同步读出，因此地址、象限和有效标志做两级流水，保证 sample 与象限匹配。
    always @(posedge clk) begin
        if (reset) begin
            phase_acc <= 22'd0;
            rom_addr <= 10'd0;
            area_stage1 <= 2'd0;
            area_stage2 <= 2'd0;
            valid_stage1 <= 1'b0;
            valid_stage2 <= 1'b0;
            new_sample_ready <= 1'b0;
            sample <= 16'd0;
        end else begin
            new_sample_ready <= valid_stage2;
            valid_stage2 <= valid_stage1;
            valid_stage1 <= sampling_pulse;
            area_stage2 <= area_stage1;

            if (valid_stage2) begin
                case (area_stage2)
                    2'd0,
                    2'd1: sample <= raw_data;
                    2'd2,
                    2'd3: sample <= ~raw_data + 1'b1;
                endcase
            end

            if (sampling_pulse) begin
                phase_acc <= next_phase;
                area_stage1 <= sine_index[11:10];
                // 只存储 1/4 周期正弦表，其余象限通过地址镜像和补码取反得到。
                case (sine_index[11:10])
                    2'd0,
                    2'd2: rom_addr <= index_low;
                    2'd1,
                    2'd3: rom_addr <= (index_low == 10'd0) ? 10'd1023 : (10'd1024 - index_low);
                endcase
            end
        end
    end
endmodule

