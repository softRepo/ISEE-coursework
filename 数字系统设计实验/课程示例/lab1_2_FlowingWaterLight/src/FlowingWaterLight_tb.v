`timescale 1ns/100ps
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
module FlowingWateLight_tb;
parameter DELY=10;
reg	clk, reset, direction;
wire [7:0] 	led;
//椤跺眰妯″潡瀹炰緥锛屼紶閫抯im鍙傛暟1
FlowingWaterLight #(.sim(1)) FlowingWateLight_inst(
    .clk(clk),.reset(reset),.direction(direction),.led (led));
//浜х敓鏃堕挓娉㈠舰
initial begin clk = 0;	forever #5 clk = ~clk;end
///浜х敓reset銆?direction娉㈠舰
initial
  begin
   reset = 0; direction = 1;
   #(DELY/2+1)reset = 1;
   #(DELY)reset = 0;
   #(DELY*80) direction = 0;
   #(DELY*90)$stop;
end
endmodule















