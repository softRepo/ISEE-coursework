module ModeComparator(a, b, m, y);
    input [7:0] a, b;
    input m;
    output [7:0] y;
    // 实例化数据比较器，通过参数传递设置为 8 位
    wire agb;
    comp #(.n(8)) comp_inst(.a(a), .b(b), .agb(agb), .aeb(), .alb());
    // 生成 addr 控制信号，逻辑为 m 与 agb 同或
    wire addr;
    assign addr = m ^~ agb; //addr产生电路就是一个同或门
    // 实例化二选一数据选择器，通过参数传递设置为 8 位
    mux_2to1 #(.n(8)) mux_inst(.out(y), .in0(a), .in1(b), .addr(addr));
endmodule
