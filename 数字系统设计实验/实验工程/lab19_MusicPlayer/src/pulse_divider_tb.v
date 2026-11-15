`timescale 1ns / 1ps

// pulse_divider 测试平台：使用较小分频比 n=8，便于在仿真中观察输出节拍。
module pulse_divider_tb_v;
    parameter delay = 10;

    reg clk;
    reg reset;
    reg enable;
    wire pulse;

    integer enable_count;
    integer pulse_count;

    pulse_divider #(
        .n(8),
        .counter_bits(3)
    ) uut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .pulse(pulse)
    );

    initial begin
        clk = 1'b0;
        reset = 1'b1;
        enable = 1'b0;
        enable_count = 0;
        pulse_count = 0;

        #(delay * 2 + 1);
        reset = 1'b0;

        // 连续输入 24 个 enable 脉冲，期望每 8 个输入产生一次 pulse。
        repeat (24) begin
            #(delay) enable = 1'b1;
            #(delay) enable = 1'b0;
        end

        // 暂停 enable，确认没有输入脉冲时输出也不会继续跳变。
        #(delay * 8);

        // 再输入 8 个 enable 脉冲，期望再次产生一个 pulse。
        repeat (8) begin
            #(delay) enable = 1'b1;
            #(delay) enable = 1'b0;
        end

        #(delay * 4);
        $display("pulse_divider_tb: enable_count=%0d, pulse_count=%0d", enable_count, pulse_count);
        $stop;
    end

    always #(delay / 2) clk = ~clk;

    always @(posedge clk) begin
        if (reset) begin
            enable_count <= 0;
            pulse_count <= 0;
        end else begin
            if (enable) begin
                enable_count <= enable_count + 1;
            end
            if (pulse) begin
                pulse_count <= pulse_count + 1;
                $display("pulse_divider_tb: pulse at %0t after enable_count=%0d", $time, enable_count);
            end
        end
    end
endmodule

