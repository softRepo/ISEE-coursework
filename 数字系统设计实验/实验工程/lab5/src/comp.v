module comp (a, b, agb, aeb, alb);
  parameter n = 1; //参数表示比较器的位数
  input [n - 1 : 0] a, b;
  output agb; //a大于b
  output aeb; //a等于b
  output alb; //a小于b
  // 利用关系运算符进行连续赋值
  assign agb = (a > b);
  assign aeb = (a == b);
  assign alb = (a < b);
endmodule
