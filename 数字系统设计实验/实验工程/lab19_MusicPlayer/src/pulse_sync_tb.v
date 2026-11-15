`timescale 1ns / 1ps

// pulse_sync 测试平台：输入不同宽度、不同相位的脉冲，观察输出是否为 clk 同步单周期脉冲。
module pulse_sync_tb_v;
    parameter delay = 10;

    reg clk;
    reg reset;
    reg in;
    wire out;

    integer out_count;

    pulse_sync uut (
        .clk(clk),
        .reset(reset),
        .in(in),
        .out(out)
    );

    initial begin
        clk = 1'b0;
        reset = 1'b1;
        in = 1'b0;
        out_count = 0;

        #(delay * 2 + 3);
        reset = 1'b0;

        // 输入一个宽脉冲，输出应只保持一个 clk 周期。
        #(delay * 3 + 2) in = 1'b1;
        #(delay * 3)     in = 1'b0;

        // 输入一个与 clk 不对齐的窄脉冲，用于观察同步后的延迟。
        #(delay * 5 + 1) in = 1'b1;
        #(delay * 2)     in = 1'b0;

        // 再输入一个宽脉冲，验证模块可以连续工作。
        #(delay * 4 + 3) in = 1'b1;
        #(delay * 4)     in = 1'b0;

        #(delay * 8);
        $display("pulse_sync_tb: out_count=%0d", out_count);
        $stop;
    end

    always #(delay / 2) clk = ~clk;

    always @(posedge clk) begin
        if (reset) begin
            out_count <= 0;
        end else if (out) begin
            out_count <= out_count + 1;
            $display("pulse_sync_tb: synchronized pulse at %0t", $time);
        end
    end
endmodule

