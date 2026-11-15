module rsff (
    input clk,  // 时钟信号
    input r,    // 同步复位（清零）信号
    input s,    // 同步置位信号
    output reg q // 输出信号
);
    // 状态更新仅发生在时钟的上升沿
    always @(posedge clk) begin
        if (r) begin
            q <= 1'b0;  // 如果 r 为 1，则清零。优先级最高
        end
        else if (s) begin
            q <= 1'b1;  // 如果 r 为 0 且 s 为 1，则置 1
        end
    end

endmodule
