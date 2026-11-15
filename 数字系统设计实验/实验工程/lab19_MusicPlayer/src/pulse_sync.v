`timescale 1ns / 1ps

// 脉冲同步模块：把异步或其他时钟域的脉冲转换为当前 clk 域内一个周期宽度的脉冲。
module pulse_sync (
    input  wire clk,
    input  wire reset,
    input  wire in,
    output wire out
);
    reg sync_0;
    reg sync_1;
    reg sync_2;

    // 两级触发器完成同步，第三级用于上升沿检测，输出单周期同步脉冲。
    always @(posedge clk) begin
        if (reset) begin
            sync_0 <= 1'b0;
            sync_1 <= 1'b0;
            sync_2 <= 1'b0;
        end else begin
            sync_0 <= in;
            sync_1 <= sync_0;
            sync_2 <= sync_1;
        end
    end

    assign out = sync_1 & ~sync_2;
endmodule

