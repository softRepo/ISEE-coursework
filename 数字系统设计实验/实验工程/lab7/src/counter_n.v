module counter_n #(
    parameter n = 2,               // 参数2：分频比 (模数 n)，默认为 2
    parameter counter_bits = 1     // 参数1：计数器的位数，默认为 1
)(
    input clk,                     // 时钟信号
    input r,                       // 同步清零信号
    input en,                      // 使能信号
    output reg [counter_bits-1:0] q = 0, // 计数值输出
    output co                      // 进位输出 (分频后的脉冲)
);
    // 时序逻辑：处理计数器的状态转换 (q* = (q+1) mod n)
    always @(posedge clk) begin
        if (r) begin
            q <= {counter_bits{1'b0}}; // 同步清零
        end
        else if (en) begin
            if (q == n - 1) begin
                q <= {counter_bits{1'b0}}; // 计到 n-1 时，下一个时钟周期清零（实现模 n 计数）
            end
            else begin
                q <= q + 1'b1;             // 否则正常加 1
            end
        end
        // 如果 r=0 且 en=0，q 保持不变
    end
    assign co = en && (q == n - 1);
endmodule
