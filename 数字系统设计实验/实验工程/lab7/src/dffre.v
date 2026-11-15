module dffre #(parameter n = 1) ( // 定义参数 n，默认值为 1
    input clk,                    // 时钟信号
    input r,                      // 同步清零信号 (高电平有效)
    input en,                     // 同步使能信号 (高电平有效)
    input [n-1:0] d,              // n 位数据输入
    output reg [n-1:0] q          // n 位数据输出，必须为 reg 类型
);
    always @(posedge clk) begin
        if (r) begin
            q <= {n{1'b0}};       // 优先级最高：同步清零。
        end
        else if (en) begin
            q <= d;               // 优先级次之：使能置数
        end
        // 隐式 else 分支：若 r=0 且 en=0，则 q 保持不变 (q <= q)
    end
endmodule
