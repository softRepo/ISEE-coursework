`timescale 1ns / 1ps

// 脉冲分频器：累计 enable 脉冲，每 n 个输入脉冲输出一个 clk 周期的 pulse。
// 实验要求将分频比设置为参数 n，将计数器位宽设置为参数 counter_bits。
module pulse_divider #(
    parameter n = 1000,
    parameter counter_bits = 10
)(
    input  wire clk,
    input  wire reset,
    input  wire enable,
    output reg  pulse
);
    reg [counter_bits-1:0] count;

    // 只在 enable 有效时计数，适合把 48 kHz ready 分频成 48 Hz beat。
    always @(posedge clk) begin
        if (reset) begin
            count <= {counter_bits{1'b0}};
            pulse <= 1'b0;
        end else begin
            pulse <= 1'b0;
            if (enable) begin
                if (count == n - 1) begin
                    count <= {counter_bits{1'b0}};
                    pulse <= 1'b1;
                end else begin
                    count <= count + 1'b1;
                end
            end
        end
    end
endmodule

