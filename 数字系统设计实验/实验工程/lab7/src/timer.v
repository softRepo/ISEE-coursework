module timer #(
    parameter n = 12,               // 默认定时时间为 12
    parameter counter_bits = 4      // 默认计数器位数为 4
)(
    input clk,                      // 时钟信号
    input r,                        // 同步清零信号 (这里接 reset || start)
    input en,                       // 使能信号
    output done                     // 定时结束标志
);
    // 内部寄存器，用于记录当前的计数值
    reg [counter_bits-1:0] q;
    // 时序逻辑：普通的累加计数
    always @(posedge clk) begin
        if (r) begin
            q <= {counter_bits{1'b0}};  // 只要有复位或重启信号，立刻清零
        end
        else if (en) begin
            q <= q + 1'b1;              // 只要使能有效，就加 1，不考虑溢出清零
        end
        // 隐式：没复位也没使能时，保持当前计数值
    end
    assign done = en && (q == n - 1);
endmodule
