module mux_2to1(out, in0, in1, addr);
    parameter n = 1;  //参数表示输入、输出数据的位数
    output [n - 1 : 0] out;
    input [n - 1 : 0] in0, in1;
    input addr;
    // 根据 addr 的值选择输出 in1 或 in0
    assign out = addr ? in1 : in0;
endmodule
