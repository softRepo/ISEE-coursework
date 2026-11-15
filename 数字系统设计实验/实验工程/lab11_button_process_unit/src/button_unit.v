`timescale 1ns / 1ps

// 按键处理单元：先同步异步按键，再消除机械抖动，最后输出一个 clk 周期宽度的脉冲。
module button_unit #(
    parameter sim = 0
)(
    input  wire clk,
    input  wire reset,
    input  wire ButtonIn,
    output wire ButtonOut
);
    wire button_sync;
    wire button_level;

    synchronizer synchronizer_inst (
        .clk(clk),
        .reset(reset),
        .async_in(ButtonIn),
        .sync_out(button_sync)
    );

    debouncer #(.sim(sim)) debouncer_inst (
        .clk(clk),
        .reset(reset),
        .in(button_sync),
        .out(button_level)
    );

    pulse_converter pulse_converter_inst (
        .clk(clk),
        .reset(reset),
        .in(button_level),
        .out(ButtonOut)
    );
endmodule

// 兼容后续实验使用的模块名，内部复用 button_unit。
module button_process_unit #(
    parameter sim = 0
)(
    input  wire clk,
    input  wire reset,
    input  wire ButtonIn,
    output wire ButtonOut
);
    button_unit #(.sim(sim)) button_unit_inst (
        .clk(clk),
        .reset(reset),
        .ButtonIn(ButtonIn),
        .ButtonOut(ButtonOut)
    );
endmodule

module synchronizer (
    input  wire clk,
    input  wire reset,
    input  wire async_in,
    output wire sync_out
);
    // 两级触发器同步异步输入，降低亚稳态继续传播的概率。
    reg sync_0;
    reg sync_1;

    always @(posedge clk) begin
        if (reset) begin
            sync_0 <= 1'b0;
            sync_1 <= 1'b0;
        end else begin
            sync_0 <= async_in;
            sync_1 <= sync_0;
        end
    end

    assign sync_out = sync_1;
endmodule

module debouncer #(
    parameter sim = 0
)(
    input  wire clk,
    input  wire reset,
    input  wire in,
    output reg  out
);
    // sim=1 时缩小分频比，便于 testbench 在较短仿真时间内观察到防抖结果。
    localparam DIVIDER = sim ? 32 : 100000;
    localparam DIVIDER_BITS = sim ? 5 : 17;

    localparam LOW       = 2'd0;
    localparam WAIT_HIGH = 2'd1;
    localparam HIGH      = 2'd2;
    localparam WAIT_LOW  = 2'd3;

    reg [1:0] state;
    reg [DIVIDER_BITS-1:0] divider_count;
    reg [3:0] ms_count;
    wire tick_1khz;
    wire timer_done;

    assign tick_1khz = (divider_count == DIVIDER - 1);
    assign timer_done = tick_1khz && (ms_count == 4'd9);

    // 产生约 1 kHz 的计时基准；仿真模式下为加速后的等效基准。
    always @(posedge clk) begin
        if (reset) begin
            divider_count <= {DIVIDER_BITS{1'b0}};
        end else if (tick_1khz) begin
            divider_count <= {DIVIDER_BITS{1'b0}};
        end else begin
            divider_count <= divider_count + 1'b1;
        end
    end

    // 四状态防抖状态机：按下和释放都必须稳定约 10 ms 后才更新输出。
    always @(posedge clk) begin
        if (reset) begin
            state <= LOW;
            out <= 1'b0;
            ms_count <= 4'd0;
        end else begin
            case (state)
                LOW: begin
                    out <= 1'b0;
                    ms_count <= 4'd0;
                    if (in) begin
                        state <= WAIT_HIGH;
                    end
                end

                WAIT_HIGH: begin
                    if (!in) begin
                        state <= LOW;
                        ms_count <= 4'd0;
                    end else if (timer_done) begin
                        state <= HIGH;
                        out <= 1'b1;
                        ms_count <= 4'd0;
                    end else if (tick_1khz) begin
                        ms_count <= ms_count + 1'b1;
                    end
                end

                HIGH: begin
                    out <= 1'b1;
                    ms_count <= 4'd0;
                    if (!in) begin
                        state <= WAIT_LOW;
                    end
                end

                WAIT_LOW: begin
                    if (in) begin
                        state <= HIGH;
                        ms_count <= 4'd0;
                    end else if (timer_done) begin
                        state <= LOW;
                        out <= 1'b0;
                        ms_count <= 4'd0;
                    end else if (tick_1khz) begin
                        ms_count <= ms_count + 1'b1;
                    end
                end

                default: begin
                    state <= LOW;
                    out <= 1'b0;
                    ms_count <= 4'd0;
                end
            endcase
        end
    end
endmodule

module pulse_converter (
    input  wire clk,
    input  wire reset,
    input  wire in,
    output wire out
);
    // 检测稳定按键信号的上升沿，只在按下瞬间输出一个时钟周期的高电平。
    reg in_dly;

    always @(posedge clk) begin
        if (reset) begin
            in_dly <= 1'b0;
        end else begin
            in_dly <= in;
        end
    end

    assign out = in & ~in_dly;
endmodule

