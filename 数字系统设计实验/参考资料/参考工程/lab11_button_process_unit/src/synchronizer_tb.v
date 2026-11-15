`timescale 1ns / 10ps

module synchronizer_tb;
    reg in, clk;
    wire out;
    parameter delay=10;

    initial begin
      clk = 0;
      in = 0;
      #(delay*100 )
        repeat (25)
          begin
            #(delay*5) in=0;
            #(delay*5) in=1;
          end
        #(delay*100)
        $stop;
      end
 //
    always #(delay/2) clk=~clk;

synchronizer synchronizer(
    .clk(clk),
    .in(in),
    .out(out)
    );
endmodule
